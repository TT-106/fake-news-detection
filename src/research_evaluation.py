"""按声明计算的评估；不对论文假设的胜负作测试断言。"""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score

from src.core_models import LABELS
from src.learned_components import BranchObservation, chow_loss
from src.research_contracts import ArchivedPassage, Prediction, ProtocolError, fingerprint, normalized_url


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    if not y_true or len(y_true) != len(y_pred) or any(y not in LABELS for y in y_true):
        raise ProtocolError("评价输入为空、长度不同或含未知真值")
    if any(y not in LABELS+["ABSTAIN", "UNAVAILABLE"] for y in y_pred):
        raise ProtocolError("未知预测标签")
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {"macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "accuracy": sum(a == b for a, b in zip(y_true, y_pred))/len(y_true),
            "per_class": {label: report[label] for label in LABELS}}


def summarize_predictions(predictions: dict[str, Prediction], truths: dict[str, str], costs: dict[str, float]) -> dict:
    if not predictions or set(predictions) != set(truths) or set(costs) != set(truths):
        raise ProtocolError("预测、真值、费用的声明 ID 必须一一匹配")
    if any(not math.isfinite(c) or c < 0 for c in costs.values()):
        raise ProtocolError("费用缺失或无效")
    ids = sorted(truths)
    actual, final = [truths[i] for i in ids], [predictions[i].final_label for i in ids]
    forced = [predictions[i].raw_label for i in ids]
    covered = [i for i in ids if not predictions[i].abstained]
    all_metrics = classification_metrics(actual, final)
    forced_metrics = classification_metrics(actual, forced)
    selective = classification_metrics([truths[i] for i in covered], [predictions[i].final_label for i in covered]) if covered else None
    return {"num_claims": len(ids), "chow_risk": float(np.mean([chow_loss(predictions[i], truths[i]) for i in ids])),
            "macro_f1": all_metrics["macro_f1"], "per_class": all_metrics["per_class"],
            "forced_macro_f1": forced_metrics["macro_f1"], "forced_accuracy": forced_metrics["accuracy"],
            "coverage": len(covered)/len(ids),
            "selective_risk": None if selective is None else 1-selective["accuracy"],
            "selective_macro_f1": None if selective is None else selective["macro_f1"],
            "mean_actual_cost_rmb": math.fsum(costs.values())/len(ids)}


def paired_bootstrap(left: dict[str, float], right: dict[str, float], seed: int = 42, n_boot: int = 4000,
                     event_groups: dict[str, str] | None = None) -> dict:
    if not left or set(left) != set(right) or n_boot < 1:
        raise ProtocolError("配对区间需要相同的非空声明 ID 集合")
    ids = sorted(left)
    differences = np.asarray([left[k]-right[k] for k in ids], dtype=float)
    if not np.all(np.isfinite(differences)):
        raise ProtocolError("配对输入存在非有限数")
    rng = np.random.default_rng(seed)
    if event_groups is None:
        boot = np.asarray([np.mean(rng.choice(differences, len(differences), replace=True)) for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"mean_difference": float(np.mean(differences)), "95ci": [float(lo), float(hi)],
                "resampling_unit": "claim", "num_pairs": len(differences), "seed": seed}
    clusters: dict[str, list[float]] = {}
    for cid, diff in zip(ids, differences.tolist()):
        gid = event_groups.get(cid)
        if not isinstance(gid, str) or not gid.strip():
            raise ProtocolError(f"组级配对区间缺少有效事件组编号：{cid}")
        clusters.setdefault(gid, []).append(diff)
    ordered = sorted(clusters)
    cluster_sums = np.asarray([math.fsum(clusters[g]) for g in ordered], dtype=float)
    cluster_sizes = np.asarray([len(clusters[g]) for g in ordered], dtype=float)
    cluster_means = cluster_sums / cluster_sizes
    n_clusters = len(ordered)
    # 整组有放回抽样并用比值估计量对齐全样本按声明均值；单条一组时退化为标准声明级 Bootstrap。
    indices = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    boot = cluster_sums[indices].sum(axis=1) / cluster_sizes[indices].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_difference": float(np.mean(differences)),
            "mean_difference_cluster_equal_weight": float(np.mean(cluster_means)),
            "95ci": [float(lo), float(hi)],
            "resampling_unit": "event_group",
            "num_pairs": len(differences),
            "num_clusters": n_clusters,
            "max_cluster_size": int(np.max(cluster_sizes)),
            "seed": seed}


def fixed_coverage_risk(predictions: dict[str, Prediction], truths: dict[str, str], coverage: float) -> dict:
    if not 0 < coverage <= 1 or not predictions or set(predictions) != set(truths):
        raise ProtocolError("固定覆盖率评价输入无效")
    # 仅用于预注册覆盖率上的 E2 风险比较，不据此修改部署的 0.75 门控。
    ranked = sorted(predictions, key=lambda cid: (-max(predictions[cid].probabilities), cid))
    eligible = [i for i in ranked if predictions[i].citations_replayable]
    k = math.ceil(len(ranked)*coverage)
    if len(eligible) < k:
        return {"status": "INSUFFICIENT_CITABLE_COVERAGE", "requested_coverage": coverage,
                "max_citable_coverage": len(eligible)/len(ranked), "risk": None}
    selected = eligible[:k]
    return {"status": "EVALUATED", "requested_coverage": coverage, "actual_coverage": k/len(ranked),
            "risk": sum(predictions[i].raw_label != truths[i] for i in selected)/k,
            "selected_claim_ids": selected}


def action_regrets(rows: list[BranchObservation], chosen: dict[tuple[str, str], str], gamma: float = 1.) -> dict:
    histories = {}
    for r in rows:
        histories.setdefault((r.claim_id, r.history_id), {})[r.action_id] = r
    if set(histories) != set(chosen):
        raise ProtocolError("行动选择必须与每条共同历史一一对应")
    regrets = {}
    for key, actions in histories.items():
        if "STOP" not in actions or chosen[key] not in actions:
            raise ProtocolError("行动遗憾必须包含零收益 STOP 与实际选定行动")
        utility = {aid: r.gain-gamma*r.realized_cost for aid, r in actions.items()}
        if utility["STOP"] != 0:
            raise ProtocolError("STOP 净收益必须严格为零")
        regrets[key] = max(utility.values())-utility[chosen[key]]
    # 多历史声明先在声明内平均，避免分支更多的声明获得额外统计权重。
    by_claim = {}
    for (cid, _), regret in regrets.items():
        by_claim.setdefault(cid, []).append(regret)
    per_claim = {cid: float(np.mean(values)) for cid, values in by_claim.items()}
    return {"mean_simple_regret": float(np.mean(list(per_claim.values()))), "per_claim": per_claim,
            "num_histories": len(histories), "num_claims": len(by_claim)}


def _blind_citation(citation, archive_dir):
    """字段白名单重建引文，模型/方法路径只进入私钥，不递归透传任意元数据。"""
    if isinstance(citation, str):
        # 历史仅 ID 的调用可继续生成空审计表，但不冒充已归档的可回放材料。
        return {"archive_id": "unresolved-"+fingerprint(citation), "replay_status": "UNAVAILABLE_ID_ONLY"}, {"original_evidence_id": citation}
    if not isinstance(citation, dict):
        raise ProtocolError("盲审引文须为归档片段对象")
    allowed = {f.name for f in fields(ArchivedPassage)}
    values = {k: v for k,v in citation.items() if k in allowed}
    for key, value in values.items():
        if key in {"start", "end"}:
            valid = type(value) is int
        elif key == "image_provenance_verified":
            valid = type(value) is bool
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ProtocolError(f"盲审引文字段不得携带嵌套元数据：{key}")
    try:
        passage = ArchivedPassage(**values)
        passage.verify()
    except (TypeError, OSError) as exc:
        raise ProtocolError("盲审引文缺少可核验的归档材料") from exc
    public = {key: getattr(passage, key) for key in
              ("url", "snapshot_sha256", "start", "end", "text", "retrieved_at", "published_at", "author")}
    public["archive_id"] = "archive-"+fingerprint({"snapshot": passage.snapshot_sha256,
                                                  "url": normalized_url(passage.url),
                                                  "start": passage.start, "end": passage.end})
    public["replay_status"] = "ORIGINAL_VERIFIED_NO_NEUTRAL_COPY"
    if archive_dir is not None:
        root = Path(archive_dir)
        name = f"snapshots/{passage.snapshot_sha256}.txt"
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = Path(passage.snapshot_path).read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest() != passage.snapshot_sha256:
            raise ProtocolError("复制盲审快照时原文件发生变化")
        if target.exists() and target.read_bytes() != raw:
            raise ProtocolError("中性盲审快照已有不同内容，拒绝覆盖")
        if not target.exists():
            target.write_bytes(raw)
        public.update(archive_file=name, replay_status="NEUTRAL_ARCHIVE_AVAILABLE")
    private = {"original_evidence_id": passage.evidence_id, "original_snapshot_path": passage.snapshot_path,
               "provenance_proof": passage.provenance_proof}
    return public, private


def replay_blind_citation(citation: dict, archive_dir) -> ArchivedPassage:
    """从中性归档复核原始正文及准确片段位置，不需私钥或方法目录。"""
    sha = citation.get("snapshot_sha256", "")
    if (not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
            or citation.get("archive_file") != f"snapshots/{sha}.txt"):
        raise ProtocolError("中性引文归档标识无效或无可回放副本")
    passage = ArchivedPassage(citation["archive_id"], citation["url"],
                               str(Path(archive_dir)/citation["archive_file"]), sha,
                               citation["start"], citation["end"], citation["text"],
                               citation["retrieved_at"], citation.get("published_at", ""), citation.get("author", ""))
    passage.verify()
    return passage


def blind_audit_packages(method_a: dict, method_b: dict, count: int = 60, seed: int = 42,
                         archive_dir=None) -> tuple[list, dict]:
    if set(method_a) != set(method_b) or count > len(method_a) or count < 1:
        raise ProtocolError("双盲配对审计必须使用同一声明集合并预先确定抽样数量")
    rng = np.random.default_rng(seed)
    ids = rng.choice(sorted(method_a), count, replace=False).tolist()
    packages, private_key = [], {}
    for cid in ids:
        sides = [("method_a", method_a[cid]), ("method_b", method_b[cid])]
        rng.shuffle(sides)
        for method, evidence in sides:
            anonymous = f"package-{int(rng.integers(0, 2**63)):016x}"
            if evidence.get("decision") not in LABELS+["ABSTAIN"] or not isinstance(evidence.get("citations"), (list, tuple)):
                raise ProtocolError("盲审包必须包含合法判定及引文列表")
            projected = [_blind_citation(c, archive_dir) for c in evidence["citations"]]
            private_key[anonymous] = {"claim_id": cid, "method": method,
                                      "citation_mapping": [private for _,private in projected]}
            # 仅复制引文包白名单，禁止方法名、自动充分性分数和自动正确性进入人工审计包。
            packages.append({"package_id": anonymous, "claim_id": cid,
                             "decision": evidence["decision"], "citations": [public for public,_ in projected],
                             "human_review_a": None, "human_review_b": None, "adjudication": None})
    rng.shuffle(packages)
    return packages, private_key
