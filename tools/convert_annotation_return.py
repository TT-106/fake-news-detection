"""将经双人独立审核及必要第三人仲裁的人工回收，严格验真并转换为训练/评测规范 JSONL。

严格边界：
- 先调用 `tools/validate_annotation_return.py::compare_returns` 校验双人独立返回与包清单。
- 数值 features、原声明 ID 与 ArchivedPassage 证明须来自协调者侧真实程序生成的 `--task-objects`，绝不编造特征或复用数据集真假标签。
- 未完成、缺材料或未仲裁分歧绝不写入训练 JSONL；`unable_to_verify`（None）永不转为负例 0。
- 对每条导出记录实时执行 `ReviewedLabel.resolve()`、`ArchivedPassage.verify()`、维度校验与划分用途核对（训练仅限 `module_fit`，来源评测仅限 `mechanism_validation`，严禁 `final_eval`）。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learned_components import ExperimentSplit, ReviewedLabel  # noqa: E402
from src.data_integrity import event_split_approved  # noqa: E402
from src.research_contracts import ArchivedPassage, ProtocolError, fingerprint  # noqa: E402
from tools.validate_annotation_return import (  # noqa: E402
    ValidationError,
    compare_returns,
    read_json,
)

FEATURE_DIMS = {"source": 6, "quality": 3, "sufficiency": 7}
ALLOWED_ROLES = {
    "source": "module_fit",
    "quality": "module_fit",
    "sufficiency": "module_fit",
    "source_eval": "mechanism_validation",
}


def _norm_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name).strip().casefold()


def _validate_adjudication_record(task_id: str, rec: dict, reviewer_a: str, reviewer_b: str) -> tuple[str, int | None, str]:
    if not isinstance(rec, dict):
        raise ProtocolError(f"仲裁记录须为对象：{task_id}")
    adjudicator = rec.get("adjudicator")
    if not isinstance(adjudicator, str) or not adjudicator.strip():
        raise ProtocolError(f"仲裁记录缺少第三人身份：{task_id}")
    adjudicator = adjudicator.strip()
    if _norm_name(adjudicator) in {_norm_name(reviewer_a), _norm_name(reviewer_b)}:
        raise ProtocolError(f"仲裁者不能与甲或乙为同一人：{task_id}")
    adj_label = rec.get("adjudicated_label")
    if adj_label is not None and (type(adj_label) is not int or adj_label not in (0, 1)):
        raise ProtocolError(f"仲裁标签只允许 0、1 或 null：{task_id}")
    rationale = rec.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ProtocolError(f"仲裁记录须含非空依据：{task_id}")
    return adjudicator, adj_label, rationale.strip()


def _verify_proofs(proofs: list[dict], expected_refs: set[str], task_id: str) -> list[dict]:
    if not isinstance(proofs, list) or not proofs:
        raise ProtocolError(f"任务缺少实际归档证据对象：{task_id}")
    verified = []
    available_ids = set()
    for raw in proofs:
        if not isinstance(raw, dict):
            raise ProtocolError(f"证据证明须为 ArchivedPassage 字段对象：{task_id}")
        passage = ArchivedPassage(**raw)
        passage.verify()
        verified.append(asdict(passage))
        available_ids.add(passage.evidence_id)
    if not expected_refs <= available_ids:
        raise ProtocolError(f"审核员引用的 evidence_refs 与归档证据对象不匹配：{task_id}")
    return verified


def convert_returns(
    return_a: dict,
    return_b: dict,
    manifest_a: dict,
    manifest_b: dict,
    task_objects: dict,
    split_data: dict,
    adjudications: dict | None = None,
    *,
    allow_provisional_split: bool = False,
    strict_complete: bool = False,
) -> dict:
    """Validate and convert dual returns + fixed task objects into canonical JSONL payloads."""
    comparison = compare_returns(return_a, return_b, manifest_a, manifest_b)
    reviewer_a = return_a["reviewer_name"].strip()
    reviewer_b = return_b["reviewer_name"].strip()
    if _norm_name(reviewer_a) == _norm_name(reviewer_b):
        raise ValidationError("两名审核者姓名规范化后相同，拒绝转换")

    if not allow_provisional_split and not event_split_approved(split_data, ROOT):
        raise ProtocolError("事件划分尚未完成人工复核（human_event_review_complete!=true），拒绝导出正式训练文件")

    split = ExperimentSplit(**{k: frozenset(v) for k, v in split_data["split"].items()})
    event_groups = split_data["event_groups"]
    split.validate(event_groups)
    role_map = {cid: role for role, cids in asdict(split).items() for cid in cids}

    idx_a = {r["task_id"]: r for r in return_a["annotation_reviews"]}
    idx_b = {r["task_id"]: r for r in return_b["annotation_reviews"]}
    adjudications = adjudications or {}

    if not isinstance(task_objects, dict) or not isinstance(task_objects.get("tasks"), dict):
        raise ProtocolError("task_objects 须包含 tasks 映射")
    tasks_meta = task_objects["tasks"]
    bound_package = None
    if "materials_sha256" in manifest_a or "materials_sha256" in manifest_b or task_objects.get("schema_version") == 2:
        if (task_objects.get("protocol") != "SPRINT2_FIXED_BLIND_V1"
                or task_objects.get("split_content_sha256") != fingerprint(split_data)):
            raise ProtocolError("回收包与固定对象、正式划分的绑定不一致")
        bound_package = task_objects.get("packages", {}).get(manifest_a["package_id"])
        if not bound_package or any(m.get("materials_sha256") != bound_package["materials_sha256"]
                or m["expected_task_ids"] != bound_package["expected_task_ids"] for m in (manifest_a, manifest_b)):
            raise ProtocolError("甲乙材料摘要或任务清单与协调者归档不符")
        for tid in bound_package["expected_task_ids"]:
            meta = tasks_meta.get(tid)
            if not meta or fingerprint(meta) != bound_package["object_sha256_by_id"].get(tid):
                raise ProtocolError(f"固定特征或证据证明的任务摘要不符：{tid}")
            kind = "source" if meta["export_kind"] == "source_eval" else meta["export_kind"]
            if kind != manifest_a["task_kind_by_id"][tid] or meta["materials_ready"] != manifest_a["task_materials_ready_by_id"][tid]:
                raise ProtocolError(f"公开任务类型或材料状态与固定对象不符：{tid}")

    outputs: dict[str, list[dict]] = {k: [] for k in ALLOWED_ROLES}
    skipped = {
        "missing_submission": [],
        "pending_materials": [],
        "temperature_calibration_excluded": [],
        "unresolved_disagreement": [],
        "missing_task_object": [],
    }

    for task_id in manifest_a["expected_task_ids"]:
        kind_manifest = manifest_a["task_kind_by_id"][task_id]
        if kind_manifest == "temperature_calibration":
            skipped["temperature_calibration_excluded"].append(task_id)
            continue

        row_a, row_b = idx_a.get(task_id), idx_b.get(task_id)
        if row_a is None or row_b is None:
            skipped["missing_submission"].append(task_id)
            continue
        if row_a["status"] == "pending_materials" or row_b["status"] == "pending_materials":
            skipped["pending_materials"].append(task_id)
            continue

        meta = tasks_meta.get(task_id)
        if meta is None:
            skipped["missing_task_object"].append(task_id)
            continue

        export_kind = meta.get("export_kind", kind_manifest)
        if export_kind not in ALLOWED_ROLES:
            raise ProtocolError(f"未知导出类别 {export_kind}：{task_id}")
        if export_kind != "source_eval" and export_kind != kind_manifest:
            raise ProtocolError(f"task_objects 类别与包 manifest 不一致：{task_id}")
        if export_kind == "source_eval" and kind_manifest != "source":
            raise ProtocolError(f"source_eval 只能来自 source 标注任务：{task_id}")

        claim_id = meta.get("claim_id")
        if claim_id not in role_map or claim_id not in event_groups:
            raise ProtocolError(f"未知或未纳入划分的声明 ID：{claim_id}（{task_id}）")
        if role_map[claim_id] == "final_eval":
            raise ProtocolError(f"严禁将官方最终评测集样本写入监督/机制验证导出：{claim_id}")
        expected_role = ALLOWED_ROLES[export_kind]
        if role_map[claim_id] != expected_role:
            raise ProtocolError(f"{export_kind} 任务声明 {claim_id} 须属于 {expected_role}，实际为 {role_map[claim_id]}")

        label_a = row_a["label"] if row_a["status"] == "done" else None
        label_b = row_b["label"] if row_b["status"] == "done" else None
        refs = set(row_a["evidence_refs"]) | set(row_b["evidence_refs"])
        if bound_package is not None:
            aliases = meta["evidence_ref_aliases"]
            if any(set(row["evidence_refs"]) != set(aliases) for row in (row_a, row_b)):
                raise ProtocolError(f"审核意见未绑定同一组固定证据编号：{task_id}")
            refs = {aliases[e] for e in refs}
        proofs = _verify_proofs(meta.get("evidence_proofs"), refs, task_id)

        combined_rationale = f"A({reviewer_a}): {row_a['reason'].strip()} | B({reviewer_b}): {row_b['reason'].strip()}"
        if label_a == label_b:
            if task_id in adjudications:
                raise ProtocolError(f"双人一致任务不应包含仲裁覆盖：{task_id}")
            reviewed = ReviewedLabel(
                reviewer_a=reviewer_a,
                reviewer_b=reviewer_b,
                label_a=label_a,
                label_b=label_b,
                rationale=combined_rationale,
                reviewer_type="human",
            )
        else:
            if task_id not in adjudications:
                skipped["unresolved_disagreement"].append(
                    {"task_id": task_id, "claim_id": claim_id, "label_a": label_a, "label_b": label_b}
                )
                continue
            adj_name, adj_label, adj_reason = _validate_adjudication_record(
                task_id, adjudications[task_id], reviewer_a, reviewer_b
            )
            reviewed = ReviewedLabel(
                reviewer_a=reviewer_a,
                reviewer_b=reviewer_b,
                label_a=label_a,
                label_b=label_b,
                rationale=f"{combined_rationale} | C({adj_name}): {adj_reason}",
                adjudicator=adj_name,
                adjudicated_label=adj_label,
                reviewer_type="human",
            )

        # Enforce canonical ReviewedLabel invariants immediately
        resolved = reviewed.resolve()

        if export_kind == "source_eval":
            ev_a, ev_b = meta.get("evidence_a"), meta.get("evidence_b")
            proof_ids = {p["evidence_id"] for p in proofs}
            if not isinstance(ev_a, str) or not isinstance(ev_b, str) or ev_a == ev_b:
                raise ProtocolError(f"source_eval 缺少有效的 evidence_a/evidence_b：{task_id}")
            if ev_a not in proof_ids or ev_b not in proof_ids:
                raise ProtocolError(f"source_eval 证据编号未包含在已校验 proofs 中：{task_id}")
            outputs["source_eval"].append({
                "task_id": task_id,
                "claim_id": claim_id,
                "event_group": event_groups[claim_id],
                "evidence_a": ev_a,
                "evidence_b": ev_b,
                "label": asdict(reviewed),
                "resolved_label": resolved,
                "proofs": proofs,
            })
        else:
            dim = FEATURE_DIMS[export_kind]
            raw_feats = meta.get("features")
            if not isinstance(raw_feats, list) or len(raw_feats) != dim:
                raise ProtocolError(f"{export_kind} 特征长度须严格为 {dim}：{task_id}")
            feats = []
            for v in raw_feats:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    raise ProtocolError(f"{export_kind} 特征包含非有限值或布尔值：{task_id}")
                feats.append(float(v))
            outputs[export_kind].append({
                "task_id": task_id,
                "claim_id": claim_id,
                "event_group": event_groups[claim_id],
                "features": feats,
                "label": asdict(reviewed),
                "resolved_label": resolved,
                "evidence_proofs": proofs,
            })

    if strict_complete and (
        skipped["missing_submission"]
        or skipped["pending_materials"]
        or skipped["unresolved_disagreement"]
        or skipped["missing_task_object"]
    ):
        raise ProtocolError(f"strict-complete 模式下存在未完成、缺材料或未仲裁项：{json.dumps(skipped, ensure_ascii=False)}")

    report = {
        "schema_version": 1,
        "package_id": return_a["package_id"],
        "reviewers": {"A": reviewer_a, "B": reviewer_b},
        "human_event_review_complete": bool(split_data.get("human_event_review_complete") is True),
        "event_review_protocol": split_data.get("protocol_id"),
        "allow_provisional_split": allow_provisional_split,
        "comparison_summary": comparison.get("summary"),
        "exported_counts": {k: len(v) for k, v in outputs.items()},
        "resolved_label_breakdown": {
            k: {
                "positive_1": sum(1 for r in v if r["resolved_label"] == 1),
                "negative_0": sum(1 for r in v if r["resolved_label"] == 0),
                "unable_to_verify_null": sum(1 for r in v if r["resolved_label"] is None),
            }
            for k, v in outputs.items()
        },
        "skipped": skipped,
    }
    return {"outputs": outputs, "report": report}


def write_conversion(result: dict, out_dir: Path, *, overwrite: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, rows in result["outputs"].items():
        if not rows:
            continue
        target = out_dir / f"{kind}.jsonl"
        if target.exists() and not overwrite:
            raise FileExistsError(f"拒绝覆盖已有标注文件（如需覆盖请显式 --overwrite）：{target}")
        # Drop helper field `resolved_label` from canonical lines? Keep task_id/resolved_label harmless or clean canonical?
        # experiment_runner._load_rows reads claim_id, event_group, features, label, evidence_proofs.
        # source_eval reader reads claim_id, evidence_a, evidence_b, label, proofs.
        with target.open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                clean = {k: v for k, v in r.items() if k != "resolved_label"}
                fh.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")
    report_path = out_dir / "conversion_report.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有报告：{report_path}")
    report_path.write_text(json.dumps(result["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path, help="审核员 A 回收 JSON")
    parser.add_argument("--b", required=True, type=Path, help="审核员 B 回收 JSON")
    parser.add_argument("--manifest-a", required=True, type=Path, help="A 包 package_manifest.json")
    parser.add_argument("--manifest-b", required=True, type=Path, help="B 包 package_manifest.json")
    parser.add_argument("--task-objects", required=True, type=Path, help="协调者侧固定证据对象与真实程序特征 JSON")
    parser.add_argument("--split", required=True, type=Path, help="已人工核验锁定的 experiment_split.json")
    parser.add_argument("--adjudications", type=Path, default=None, help="可选：第三人仲裁记录 JSON")
    parser.add_argument("--out-dir", required=True, type=Path, help="输出目录")
    parser.add_argument("--allow-provisional-split", action="store_true", help="仅限演练：允许未锁定的候选划分（禁止写入 data/annotations）")
    parser.add_argument("--strict-complete", action="store_true", help="若存在缺项、待补材料或未仲裁分歧则报错退出")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖目标目录已有文件")
    args = parser.parse_args(argv)

    if args.allow_provisional_split:
        resolved_out = args.out_dir.resolve()
        if (ROOT / "data/annotations").resolve() in (resolved_out, *resolved_out.parents):
            raise ProtocolError("演练模式禁止向 data/annotations/ 写入训练文件")

    res = convert_returns(
        read_json(args.a),
        read_json(args.b),
        read_json(args.manifest_a),
        read_json(args.manifest_b),
        read_json(args.task_objects),
        read_json(args.split),
        read_json(args.adjudications) if args.adjudications else None,
        allow_provisional_split=args.allow_provisional_split,
        strict_complete=args.strict_complete,
    )
    write_conversion(res, args.out_dir, overwrite=args.overwrite)
    print(json.dumps(res["report"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
