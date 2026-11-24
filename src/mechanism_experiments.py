"""E1 的人工来源上界与 E2 同证据包的真实输入消融。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.core_models import compute_discounted_weights, solve_convex_qp_inference, theoretical_stability_bound
from src.faithful_pipeline import ClaimAnalysis, FaithfulPipeline, source_pair_features
from src.input_views import InputView
from src.learned_components import ReviewedLabel
from src.research_contracts import ArchivedPassage, CostLedger, ProtocolError, fingerprint
from src.research_evaluation import fixed_coverage_risk


@dataclass(frozen=True)
class SourceEvalPair:
    claim_id: str
    evidence_a: str
    evidence_b: str
    label: ReviewedLabel
    proofs: tuple[ArchivedPassage, ...]


class ViewSufficiencyModel:
    """保持冻结 LR 参数；在输入消融入口切断其直接出处特征旁路。"""
    purpose = "sufficiency"

    def __init__(self, model, view):
        self.model, self.view = model, view

    @property
    def version(self):
        return fingerprint({"model": self.model.version, "view": self.view.version,
                            "provenance_feature_index": 1, "projection": "WITHHELD_ZERO"})

    def predict(self, rows):
        features = np.asarray(rows, dtype=float).copy()
        if features.ndim != 2 or features.shape[1] != 7:
            raise ProtocolError("充分性输入消融要求冻结的七维特征接口")
        if self.view.mode == "no_provenance":
            features[:, 1] = 0.
        return self.model.predict(features)


def evaluate_source_methods(states: list, pairs: list[SourceEvalPair], source_model, allowed_ids: set[str]) -> dict:
    if not states or any(s.claim.claim_id not in allowed_ids for s in states):
        raise ProtocolError("E1 只能使用预先隔离的机制验证声明")
    if {s.claim.event_group for s in states} & source_model.training_groups:
        raise ProtocolError("E1 评测事件组与来源模型训练组重叠")
    known, brier_errors = {}, []
    state_map = {s.claim.claim_id: s for s in states}
    for pair in pairs:
        if pair.claim_id not in state_map or not pair.proofs:
            raise ProtocolError("来源评测标注必须指向当前真实证据并提供独立核实证明")
        for p in pair.proofs:
            p.verify()
        y = pair.label.resolve()
        if y is None:
            continue
        state = state_map[pair.claim_id]
        lookup = {e.passage.evidence_id: e.passage for e in state.evidence}
        if pair.evidence_a == pair.evidence_b or pair.evidence_a not in lookup or pair.evidence_b not in lookup:
            raise ProtocolError("来源评测标注引用了不存在的证据对")
        key = (pair.claim_id, *sorted((pair.evidence_a, pair.evidence_b)))
        if key in known:
            raise ProtocolError("同一来源评测对存在重复或冲突标注")
        known[key] = y
        p = float(source_model.predict([source_pair_features(lookup[pair.evidence_a], lookup[pair.evidence_b])])[0])
        brier_errors.append((p-y)**2)
    records = []
    for state in states:
        n = len(state.evidence)
        if n == 0:
            records.append({"claim_id": state.claim.claim_id, "status": "NO_RETRIEVED_EVIDENCE"})
            continue
        p_hat, p_oracle = np.eye(n), np.eye(n)
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                ei, ej = state.evidence[i], state.evidence[j]
                estimate = float(source_model.predict([source_pair_features(ei.passage, ej.passage)])[0])
                p_hat[i,j] = p_hat[j,i] = estimate
                key = (state.claim.claim_id, *sorted((ei.passage.evidence_id, ej.passage.evidence_id)))
                y = known.get(key)
                if y is None and ei.fact.key == ej.fact.key:
                    missing.append([ei.passage.evidence_id, ej.passage.evidence_id])
                # 不同规范事实的关系不参与式(1)分母；保留其预测值，不把未知关系伪称人工真值。
                p_oracle[i,j] = p_oracle[j,i] = estimate if y is None else float(y)
        if missing:
            records.append({"claim_id": state.claim.claim_id, "status": "INCOMPLETE_SAME_FACT_SOURCE_ANNOTATIONS",
                            "missing_pairs": missing, "oracle_state_drift": None})
            continue
        rules = [r.rule for r in state.rules]
        oracle, _ = solve_convex_qp_inference(list(state.snippets), p_oracle, rules)
        methods = {}
        for method in ("independent", "domain_dedup", "semantic_dedup", "max_quality_single", "prob_discount"):
            z, _ = solve_convex_qp_inference(list(state.snippets), p_hat, rules, weight_mode=method)
            W = compute_discounted_weights(list(state.snippets), p_hat, mode=method)
            novel = []
            for i, snippet in enumerate(state.snippets):
                previous = [s for s in state.snippets[:i] if s.domain == snippet.domain]
                if previous and all(s.fact_id != snippet.fact_id for s in previous):
                    novel.append(float(np.any(W[:,i] > 0)))
            methods[method] = {"state_drift_l2": float(np.linalg.norm(z-oracle)),
                               "same_domain_novel_fact_retention": float(np.mean(novel)) if novel else None}
        bound = theoretical_stability_bound(list(state.snippets), p_hat, p_oracle)
        records.append({"claim_id": state.claim.claim_id, "status": "EVALUATED", "methods": methods,
                        "source_error_stability_bound": bound,
                        "bound_satisfied": methods["prob_discount"]["state_drift_l2"] <= bound+1e-8})
    return {"evidence_protocol": "REAL_ARCHIVED_PASSAGES_WITH_REVIEWED_SOURCE_PAIRS",
            "source_brier_score": float(np.mean(brier_errors)) if brier_errors else None,
            "verified_eval_pairs": len(brier_errors), "records": records,
            "unknown_relations_excluded": True,
            "interpretation": "人工来源上界评测来源概率误差；它不是证明概率折扣优于所有方法的独立真实性标签。"}


def evaluate_multimodal_ablations(pipeline: FaithfulPipeline, states: list, truths: dict,
                                 views: dict[str, InputView], coverages=(.25,.5,.75)) -> dict:
    if not states or {s.claim.claim_id for s in states} != set(truths):
        raise ProtocolError("E2 声明、真实证据包和离线标签必须一一匹配")
    required = {"full", "no_image", "no_provenance", "no_spatiotemporal", "no_rules"}
    if set(views) != required or any(view.mode != name for name,view in views.items()):
        raise ProtocolError("E2 必须预先定义完整输入和四项消融视图")
    if not hasattr(pipeline.vlm, "with_view"):
        raise ProtocolError("VLM 未实现实际输入遮罩；不能用乘法降低质量分数冒充模态消融")
    results, receipts = {}, []
    for mode, view in views.items():
        model = pipeline.vlm.with_view(view)
        variant = FaithfulPipeline(model, pipeline.tools, pipeline.source, pipeline.quality,
                                   ViewSufficiencyModel(pipeline.sufficiency, view), pipeline.temperature,
                                   pipeline.sufficiency_threshold, pipeline.output_reserve)
        predictions = {}
        for state in states:
            ledger = CostLedger()
            trace_start = len(view.render_receipts)
            passages = tuple(e.passage for e in state.evidence)
            # 原始网页、顺序和原子声明固定；重新执行同一 VLM 对可见输入的真实立场/质量特征推理。
            scored = model.score(state.claim, state.atoms, passages, ledger) if mode != "no_rules" else state.evidence
            if len(scored) != len(state.evidence) or any(a.passage != b.passage for a,b in zip(scored,state.evidence)):
                raise ProtocolError("E2 不得改变各方法看到的原始证据包")
            scored = tuple(replace(a, fact=b.fact) for a,b in zip(scored,state.evidence))
            analysis = ClaimAnalysis(state.atoms, state.gaps, () if mode == "no_rules" else state.rules)
            # 来源结构和规范事实保持固定，仅比较输入/规则消融对状态与终端的影响。
            rel = {tuple(sorted((state.evidence[i].passage.evidence_id, state.evidence[j].passage.evidence_id))):
                       state.source_matrix[i][j] for i in range(len(passages)) for j in range(i+1,len(passages))}
            changed = variant.assemble_logged(state.claim, analysis, scored, ledger, relations=rel)
            prediction = variant.predict(changed, ledger)
            predictions[state.claim.claim_id] = prediction
            receipts.append({"claim_id": state.claim.claim_id, "mode": mode, "input_view": view.version,
                             "prediction": prediction, "ledger": ledger.entries,
                             "input_projection_receipts": view.render_receipts[trace_start:],
                             "sufficiency_provenance_feature": "WITHHELD_ZERO" if mode == "no_provenance" else "OBSERVED"})
        results[mode] = {"fixed_coverage_risks": [fixed_coverage_risk(predictions, truths, c) for c in coverages]}
    return {"evidence_protocol": "SAME_ARCHIVED_EVIDENCE_REAL_VLM_INPUT_ABLATIONS", "methods": results,
            "input_view_versions": {name:v.version for name,v in views.items()}, "receipts": receipts,
            "no_image_definition": "移除原始图像像素输入；保留共同检索包中的网页文本",
            "no_provenance_definition": {"vlm_metadata": "WITHHELD", "sufficiency_provenance_feature": "WITHHELD_ZERO",
                "source_matrix": "HELD_FIXED", "fact_clusters": "HELD_FIXED",
                "scope": "检验出处输入与充分性直接出处分量；来源矩阵和规范事实固定，不宣称来源结构完全盲化"},
            "source_structure_and_fact_clusters_held_fixed": True}
