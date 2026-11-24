"""真实归档材料上的模块拟合、共同历史采集、基线锁定和最终闭环。

本入口不会下载权重、请求收费 API、读取金标准 QA 充当工具返回，或自动生成标注。
E1 人工来源上界与 E2 逐字段消融需要额外的人工来源/遮罩材料，缺失时明确不出分。
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from src.archived_tools import ArchivedEvidenceTools
from src.data_integrity import AI_EVENT_PROTOCOL, audit_data, event_split_approved
from src.faithful_pipeline import (FaithfulPipeline, action_features, choose_one_step, feasible_actions)
from src.learned_components import (ActionGainRegressor, BranchObservation, CalibrationRow,
                                    ExperimentSplit, ReviewedBinaryModel, ReviewedLabel,
                                    SharedTemperature, SupervisedRow, MatchedChiPermutation, chow_loss)
from src.research_contracts import (ArchivedPassage, CostLedger, ImageAsset, Prediction,
                                    ProtocolError, PublicClaim, fingerprint)
from src.research_evaluation import (action_regrets, blind_audit_packages, paired_bootstrap,
                                    summarize_predictions)
from src.vlm_backend import LocalQwenVLM
from src.input_views import InputView, TextMask, ImageMask
from src.mechanism_experiments import SourceEvalPair, evaluate_source_methods, evaluate_multimodal_ablations


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def _load_rows(path: Path, dimension: int):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        proofs = data.get("evidence_proofs")
        if not proofs:
            raise ProtocolError("监督标注必须关联实际归档材料，不能只提供人工填写的成绩/特征表")
        for proof in proofs:
            ArchivedPassage(**proof).verify()
        features = tuple(float(v) for v in data["features"])
        if len(features) != dimension:
            raise ProtocolError("标注特征维度与冻结的特征提取器不一致")
        rows.append(SupervisedRow(data["claim_id"], data["event_group"], features, ReviewedLabel(**data["label"])))
    return rows


class RoutingBaseline:
    """公开、可审计的路由对照，不宣称复现 FIRE/DEFAME/SAFE-MR 论文成绩。"""
    def __init__(self, name):
        self.name = name
        self.version = fingerprint({"name": name, "confidence_stop": .75, "revision": 1})

    def choose(self, state, prediction, actions):
        if self.name != "static_text_once" and not prediction.abstained:
            return "STOP"
        if self.name == "static_text_once" and state.step > 0:
            return "STOP"
        order = ["image_trace", "fact_search"] if self.name == "image_first" else ["fact_search", "image_trace"]
        for kind in order:
            matches = sorted((a for a in actions if a.kind == kind), key=lambda a: a.action_id)
            if matches:
                return matches[0].action_id
        return "STOP"


def _collect(pipeline, claims, truths, budget, history_depth):
    """每个前缀从真实历史复制所有可行分支，不以标签/收益选择采样分支。"""
    rows, histories, receipt_logs = [], {}, []
    for claim in claims:
        ledger = CostLedger()
        state = pipeline.initialise(claim, ledger, budget)
        prediction = pipeline.predict(state, ledger)
        if ledger.total > pipeline.tools.initial_cost_upper_bound+1e-12:
            raise ProtocolError("共同历史的完整初态成本超过先导核算上限")
        queue = [(state, ledger, prediction)]
        while queue:
            state, ledger, prediction = queue.pop(0)
            key = (claim.claim_id, state.identity)
            if key in histories:
                continue
            spent = ledger.total
            actions = feasible_actions(state, pipeline.tools.candidates(state), spent, budget, pipeline.output_reserve)
            histories[key] = (state, spent, actions, prediction)
            loss = chow_loss(prediction, truths[claim.claim_id])
            for a in actions:
                features = action_features(state, a, spent, budget)
                branch_ledger = copy.deepcopy(ledger)
                next_state = pipeline.advance(state, a, branch_ledger, budget)
                after_prediction = pipeline.predict(next_state, branch_ledger)
                cost = branch_ledger.total-spent
                if cost > a.cost_upper_bound+1e-12:
                    raise ProtocolError("真实共同历史分支超出费用上限")
                rows.append(BranchObservation(claim.claim_id, claim.event_group, state.identity, a.action_id,
                                              features, loss, chow_loss(after_prediction, truths[claim.claim_id]),
                                              a.expected_cost, cost,
                                              (state.source_matrix[a.target_pair[0]][a.target_pair[1]] *
                                               (1-state.source_matrix[a.target_pair[0]][a.target_pair[1]])) if a.target_pair is not None else 0.))
                receipt_logs.append({"claim_id": claim.claim_id, "history_id": state.identity, "action": asdict(a),
                                     "before_prediction": asdict(prediction), "after_prediction": asdict(after_prediction),
                                     "ledger": [asdict(e) for e in branch_ledger.entries], "next_history_id": next_state.identity})
                if next_state.step < history_depth:
                    queue.append((next_state, branch_ledger, after_prediction))
            rows.append(BranchObservation(claim.claim_id, claim.event_group, state.identity,
                                          "STOP", (), loss, loss, 0., 0.))
    return rows, histories, receipt_logs


def run_experiments(config_path: Path, output_dir: Path):
    root = Path(__file__).resolve().parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_data = json.loads((root/config["experiment_split"]).read_text(encoding="utf-8"))
    if not event_split_approved(split_data, root):
        raise ProtocolError("自动候选划分尚未完成人工事件核查，不能声称严格无泄漏")
    if split_data.get("protocol_id") == AI_EVENT_PROTOCOL and config.get("unlock_final_eval") is not True:
        raise ProtocolError("此入口同时运行 E4；官方终评数据保持封存")
    split = ExperimentSplit(**{key: frozenset(value) for key, value in split_data["split"].items()})
    groups = split_data["event_groups"]
    split.validate(groups)
    if (len(split.module_fit)+len(split.temperature_calibration) != 60 or len(split.pilot) != 20
            or len(split.action_train) != 180 or len(split.mechanism_validation) != 60):
        raise ProtocolError("正式实验的开发/先导/行动训练/机制验证数量不符合预注册方案")
    audit = audit_data(root, include_final_eval=True)
    if split_data.get("dataset_sha256") != audit["dataset_sha256"]:
        raise ProtocolError("划分绑定的数据版本不匹配")
    actual_data = {}
    for name in ("train", "val"):
        actual_data.update({f"{name}:{i}": s for i,s in enumerate(json.loads((root/f"data/averimatec/{name}.json").read_text(encoding="utf-8")))})
    official = frozenset(cid for cid in actual_data if cid.startswith("val:"))
    if split.final_eval != official:
        raise ProtocolError("不得为了提高分数删减或替换官方最终评测声明")
    observed_roles = {}
    for role, ids in asdict(split).items():
        for cid in ids:
            if cid not in actual_data or cid not in audit["event_groups"]:
                raise ProtocolError("划分含未知声明")
            observed_group = audit["event_groups"][cid]
            if observed_group in observed_roles and observed_roles[observed_group] != role:
                raise ProtocolError("内容审计发现事件关联跨用途；需隔离关联训练样本")
            observed_roles[observed_group] = role
    image_dir = root/config["image_dir"]
    def claims(ids):
        result = []
        for cid in sorted(ids):
            sample = actual_data[cid]
            assets = []
            for name in sample.get("claim_images") or []:
                p = image_dir/name
                assets.append(ImageAsset(str(p.resolve()), hashlib.sha256(p.read_bytes()).hexdigest()))
            result.append(PublicClaim.from_sample(sample, cid, groups[cid], tuple(assets)))
        return result
    binary = {}
    for name, dim in (("source", 6), ("quality", 3), ("sufficiency", 7)):
        rows = _load_rows(root/config[f"{name}_annotations"], dim)
        binary[name] = ReviewedBinaryModel(name).fit(rows, split, groups)
    vlm = LocalQwenVLM(str(root/config["model_dir"]), config["quantization"], str(output_dir/"vlm_calls"),
                       int(config.get("max_pixels", 512*28*28)),
                       rule_manifest_path=str(root/config["rule_manifest"]) if config.get("rule_manifest") else None)
    tools = ArchivedEvidenceTools(str(root/config["archived_tools"]))
    temp = SharedTemperature()
    kwargs = dict(vlm=vlm, tools=tools, **binary, temperature=temp,
                  sufficiency_threshold=float(config["sufficiency_threshold"]),
                  output_reserve=float(config["output_reserve_rmb"]))
    calibration_engine = FaithfulPipeline(**kwargs, calibration_only=True)
    budget, gamma = float(config.get("budget_rmb", .25)), float(config.get("gamma", 1.))
    calibration_rows, calibration_logs = [], []
    for claim in claims(split.temperature_calibration):
        ledger = CostLedger()
        state = calibration_engine.initialise(claim, ledger, budget)
        terminal = vlm.terminal(state, ledger)
        if ledger.total > tools.initial_cost_upper_bound+1e-12:
            raise ProtocolError("校准初态的完整处理费用超过先导核算上限")
        calibration_rows.append(CalibrationRow(claim.claim_id, claim.event_group, terminal.logits,
                                               actual_data[claim.claim_id]["label"]))
        calibration_logs.append({"claim_id": claim.claim_id, "state_id": state.identity,
                                 "terminal": asdict(terminal), "ledger": [asdict(e) for e in ledger.entries]})
    temp.fit(calibration_rows, split, groups)
    pipeline = FaithfulPipeline(**kwargs)
    frozen_version = pipeline.module_version
    _write(output_dir/"module_lock.json", {"version": frozen_version, "vlm": asdict(vlm.identity),
                                           "binary_models": {name: model.version for name, model in binary.items()},
                                           "temperature": temp.temperature, "calibration": calibration_logs,
                                           "config": config, "dataset_sha256": audit["dataset_sha256"]})
    depth = int(config.get("training_history_depth", 1))
    if not 1 <= depth <= 3:
        raise ProtocolError("共同历史采集深度必须为 1—3；更深采集费用必须重新估算")
    train_truths = {cid: actual_data[cid]["label"] for cid in split.action_train}
    train_rows, _, train_receipts = _collect(pipeline, claims(split.action_train), train_truths, budget, depth)
    _write(output_dir/"action_train_branches.json", {"module_version": frozen_version,
                                                    "rows": [asdict(r) for r in train_rows], "receipts": train_receipts})
    models = {}
    for name, mode in (("full_ridge", "full"), ("direct_ridge_no_chi", "no_chi")):
        rows = train_rows if mode == "full" else [replace(r, features=r.features[:-1]) if r.action_id != "STOP" else r for r in train_rows]
        models[name] = ActionGainRegressor(mode).fit(rows, split, groups, frozen_version)
    uncertainty_rows = [replace(r, features=r.features[:-1]+(r.target_uncertainty,)) if r.action_id != "STOP" else r for r in train_rows]
    models["general_source_uncertainty"] = ActionGainRegressor("uncertainty").fit(uncertainty_rows, split, groups, frozen_version)
    permutation_status = "NOT_APPLICABLE_NO_TARGET_SOURCE_PAIRS"
    if any(r.action_id != "STOP" and r.features[-2] == 1. for r in train_rows):
        control = MatchedChiPermutation(seed=42)
        permutation_rows = control.fit_transform(train_rows)
        perm_model = ActionGainRegressor("permuted_chi").fit(permutation_rows, split, groups,
                                                           fingerprint({"modules": frozen_version, "donor_pool": control.version}))
        perm_model.permutation_control = control
        models["matched_chi_permutation"] = perm_model
        permutation_status = "FROZEN_TRAINING_UNCERTAINTY_STRATA_DONOR_POOL"
        _write(output_dir/"permutation_control_lock.json", {"version": control.version, "seed": control.seed,
                                                           "pools": control.pools, "source": "action_train_only"})
    for name in ("text_first", "image_first", "static_text_once"):
        models[name] = RoutingBaseline(name)
    # 机制验证先于官方最终评测。选出唯一竞争基线后保存锁文件。
    validation_truths = {cid: actual_data[cid]["label"] for cid in split.mechanism_validation}
    validation_rows, histories, validation_receipts = _collect(pipeline, claims(split.mechanism_validation), validation_truths, budget, depth)
    initial_states = [state for state, _, _, _ in histories.values() if state.step == 0]
    e1 = {"status": "NOT_RUN_REQUIRES_INDEPENDENT_RELATION_EVAL_ANNOTATIONS", "score": None}
    if config.get("source_eval_annotations"):
        pairs = []
        for line in (root/config["source_eval_annotations"]).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            pairs.append(SourceEvalPair(d["claim_id"], d["evidence_a"], d["evidence_b"], ReviewedLabel(**d["label"]),
                                         tuple(ArchivedPassage(**p) for p in d["proofs"])))
        e1 = evaluate_source_methods(initial_states, pairs, binary["source"], set(split.mechanism_validation))
        _write(output_dir/"e1_report.json", e1)
    e2 = {"status": "NOT_RUN_REQUIRES_VERIFIED_FIELD_MASKS", "score": None}
    if config.get("spatiotemporal_masks"):
        d = json.loads((root/config["spatiotemporal_masks"]).read_text(encoding="utf-8"))
        masks = {cid: [TextMask(**m) for m in rows] for cid,rows in d["masks"].items()}
        reviews = {cid: ReviewedLabel(**r) for cid,r in d["reviews"].items()}
        image_masks = {sha: ImageMask(**record) for sha,record in d.get("image_masks", {}).items()}
        views = {mode: InputView(mode, masks, reviews, image_masks) if mode == "no_spatiotemporal" else InputView(mode)
                 for mode in ("full", "no_image", "no_provenance", "no_spatiotemporal", "no_rules")}
        e2 = evaluate_multimodal_ablations(pipeline, initial_states, validation_truths, views)
        _write(output_dir/"e2_report.json", {**e2, "receipts": [
            {**r, "prediction": asdict(r["prediction"]), "ledger": [asdict(e) for e in r["ledger"]]}
            for r in e2["receipts"]]})
    e3 = {}
    for name, model in models.items():
        chosen = {}
        squared_errors = []
        by_outcome = {(r.claim_id,r.history_id,r.action_id):r for r in validation_rows}
        for key, (state, spent, actions, prediction) in histories.items():
            if isinstance(model, ActionGainRegressor):
                gains = {a.action_id: model.predict_gain(model.features_for(state, a, spent, budget)) for a in actions}
                chosen[key] = choose_one_step(actions, gains, gamma)
                squared_errors.extend((value-by_outcome[(*key,aid)].gain)**2 for aid,value in gains.items())
            else:
                chosen[key] = model.choose(state, prediction, actions)
        e3[name] = action_regrets(validation_rows, chosen, gamma)
        e3[name]["gain_mse"] = math.fsum(squared_errors)/len(squared_errors) if squared_errors else None
        e3[name]["positive_gain_action_hit_rate"] = sum(by_outcome[(*key,act)].gain > 0 for key,act in chosen.items())/len(chosen)
    e3_comparison = paired_bootstrap(e3["direct_ridge_no_chi"]["per_claim"], e3["full_ridge"]["per_claim"],
                                     event_groups=groups)
    _write(output_dir/"e3_report.json", {"evidence_protocol": "OPEN_WEB_ARCHIVED_REPLAY", "methods": e3,
                                         "paired_no_chi_minus_full_regret": e3_comparison,
                                         "history_depth": depth, "rows": [asdict(r) for r in validation_rows],
                                         "receipts": validation_receipts,
                                         "permutation_control": permutation_status,
                                         "uncertainty_control": "SAME_DIMENSION_TARGET_P_ONE_MINUS_P"})
    def evaluate(ids, policies, truths, run_budget=budget):
        results, episodes = {}, {}
        all_claims = claims(ids)
        for name in policies:
            episodes[name] = {c.claim_id: pipeline.run_episode(c, models[name], run_budget, gamma) for c in all_claims}
            predictions = {cid: Prediction(**ep["prediction"]) for cid,ep in episodes[name].items()}
            costs = {cid: ep["total_cost_rmb"] for cid,ep in episodes[name].items()}
            results[name] = summarize_predictions(predictions, truths, costs)
        return results, episodes
    baseline_names = [name for name in models if name != "full_ridge"]
    validation, _ = evaluate(split.mechanism_validation, baseline_names, validation_truths)
    strongest = min(baseline_names, key=lambda name: (validation[name]["chow_risk"], -validation[name]["coverage"], name))
    _write(output_dir/"baseline_lock.json", {"baseline": strongest, "validation": validation,
                                             "model_version": models[strongest].version,
                                             "selection_rule": "min mechanism-validation Chow risk; ties higher coverage then name",
                                             "final_eval_accessed_for_selection": False})
    final_truths = {cid: actual_data[cid]["label"] for cid in split.final_eval}
    results, episodes = evaluate(split.final_eval, ["full_ridge", strongest], final_truths)
    curve = [{"budget": budget, "results": results}]
    for b in sorted(set(float(v) for v in config.get("quality_cost_budgets", []))):
        if b != budget:
            curve_results, _ = evaluate(split.final_eval, ["full_ridge", strongest], final_truths, b)
            curve.append({"budget": b, "results": curve_results})
    diagnostics = {}
    for name in ("always_refuted", "always_abstain"):
        preds = {cid: Prediction((0.,1.,0.,0.), "Refuted", "Refuted" if name == "always_refuted" else "ABSTAIN",
                                 0., (), False, "DIAGNOSTIC_NO_EVIDENCE") for cid in final_truths}
        diagnostics[name] = summarize_predictions(preds, final_truths, {cid: 0. for cid in final_truths})
    loss = {name: {cid: chow_loss(Prediction(**ep["prediction"]), final_truths[cid]) for cid,ep in eps.items()}
            for name,eps in episodes.items()}
    packages = {}
    for name, eps in episodes.items():
        packages[name] = {cid: {"decision": ep["prediction"]["final_label"],
                                "citations": [asdict(e.passage) for e in ep["state"].evidence
                                              if e.passage.evidence_id in ep["prediction"]["citation_ids"]]}
                          for cid,ep in eps.items()}
    blind, private_key = blind_audit_packages(packages["full_ridge"], packages[strongest], count=min(60,len(final_truths)),
                                            archive_dir=output_dir/"human_audit_materials")
    _write(output_dir/"human_audit_packages.json", blind)
    _write(output_dir/"audit_private_key/assignment.json", private_key)
    serial_episodes = {name: {cid: {k:v for k,v in ep.items() if k != "state"} for cid,ep in eps.items()}
                       for name,eps in episodes.items()}
    report = {"protocol": "OPEN_WEB_ARCHIVED_REPLAY", "formal_vlm_inference_executed": True,
              "retrieval_execution": "MEASURED_ORIGINAL_TOOL_RECEIPTS_REPLAYED; not live-web retrieval",
              "module_version": frozen_version, "selected_baseline": strongest, "results": results,
              "diagnostic_baselines_without_evidence_gate": diagnostics,
              "paired_chow_difference": paired_bootstrap(loss[strongest], loss["full_ridge"], event_groups=groups),
              "human_evidence_audit": {"status": "AWAITING_TWO_HUMAN_REVIEWS_AND_ADJUDICATION", "score": None},
              "E1_source_upper_bound": {k:v for k,v in e1.items() if k != "records"},
              "E2_multimodal_ablations": {k:v for k,v in e2.items() if k != "receipts"},
              "quality_cost_curve": {"status": "EVALUATED_PREDECLARED_BUDGETS" if len(curve)>1 else "ONE_LOCKED_WORKING_POINT",
                                     "points": sorted(curve, key=lambda p:p["budget"])},
              "episodes": serial_episodes}
    _write(output_dir/"e4_report.json", report)
    return report
