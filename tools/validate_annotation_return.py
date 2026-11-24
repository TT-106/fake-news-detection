"""验收独立人工返回的结构并列出分歧；不仲裁、不生成训练标签或数值特征。"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

CLASS_LABELS = {"Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"}
TASK_KINDS = {"source", "quality", "sufficiency", "temperature_calibration"}
SECTIONS = {"event_reviews": ("claim_id", "expected_claim_ids"),
            "link_reviews": ("link_id", "expected_link_ids"),
            "annotation_reviews": ("task_id", "expected_task_ids")}
TOP_KEYS = {"schema_version", "package_id", "reviewer_slot", "reviewer_name", *SECTIONS}
ROW_KEYS = {"event_reviews": {"claim_id", "decision", "proposed_event_id", "merge_with", "reason", "images_reviewed", "cross_group_checked"},
            "link_reviews": {"link_id", "relation", "reason"},
            "annotation_reviews": {"task_id", "status", "label", "class_label", "reason", "evidence_refs"}}


class ValidationError(ValueError):
    def __init__(self, errors):
        self.errors = errors if isinstance(errors, list) else [str(errors)]
        super().__init__("；".join(self.errors))


def read_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValidationError(f"JSON对象键重复：{key}")
            result[key] = value
        return result
    def reject(value):
        raise ValidationError(f"JSON不能包含{value}")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs, parse_constant=reject)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"JSON语法错误，行{exc.lineno}列{exc.colno}") from exc


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValidationError("manifest必须是对象")
    errors = []
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        errors.append("manifest.schema_version须为整数1")
    if not isinstance(manifest.get("package_id"), str) or not manifest["package_id"].strip():
        errors.append("manifest.package_id不能为空")
    if not isinstance(manifest.get("reviewer_slot"), str) or manifest["reviewer_slot"] not in {"A", "B"}:
        errors.append("manifest.reviewer_slot须为A/B")
    for _, expected_key in SECTIONS.values():
        ids = manifest.get(expected_key)
        if not isinstance(ids, list) or any(not isinstance(value, str) or not value.strip() for value in ids):
            errors.append(f"manifest.{expected_key}须为非空ID组成的数组（数组本身可空）")
        elif len(ids) != len(set(ids)):
            errors.append(f"manifest.{expected_key}存在重复ID")
    if errors:
        raise ValidationError(errors)
    task_ids = set(manifest["expected_task_ids"])
    kinds, readiness = manifest.get("task_kind_by_id"), manifest.get("task_materials_ready_by_id")
    if not isinstance(kinds, dict) or set(kinds) != task_ids or any(not isinstance(kind, str) or kind not in TASK_KINDS for kind in kinds.values()):
        errors.append("manifest.task_kind_by_id须完整覆盖task IDs且为source/quality/sufficiency/temperature_calibration")
    if not isinstance(readiness, dict) or set(readiness) != task_ids or any(type(value) is not bool for value in readiness.values()):
        errors.append("manifest.task_materials_ready_by_id须完整覆盖task IDs且值严格为bool")
    if errors:
        raise ValidationError(errors)
    return manifest


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def in_enum(value, options):
    return isinstance(value, str) and value in options


def validate_return(data, manifest):
    manifest = validate_manifest(manifest)
    if not isinstance(data, dict):
        raise ValidationError("返回必须是JSON对象")
    errors = []
    if set(data) != TOP_KEYS:
        errors.append(f"返回顶层字段不匹配；缺失{sorted(TOP_KEYS-set(data))}，多出{sorted(set(data)-TOP_KEYS)}")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        errors.append("schema_version须为整数1")
    for key in ("package_id", "reviewer_slot"):
        if data.get(key) != manifest[key]:
            errors.append(f"{key}与当前包manifest不一致")
    if not nonempty(data.get("reviewer_name")) or data["reviewer_name"] != data["reviewer_name"].strip():
        errors.append("reviewer_name必须实际填写且不能仅空白或带首尾空格")
    indexes = {}
    for section, (identity, expected_key) in SECTIONS.items():
        rows = data.get(section)
        if not isinstance(rows, list):
            errors.append(f"{section}须为数组，未填用空数组")
            indexes[section] = {}
            continue
        expected, seen = set(manifest[expected_key]), set()
        indexes[section] = {}
        for index, row in enumerate(rows):
            prefix = f"{section}[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{prefix}须为对象")
                continue
            if set(row) != ROW_KEYS[section]:
                errors.append(f"{prefix}字段不匹配")
            row_id = row.get(identity)
            if not isinstance(row_id, str) or row_id not in expected:
                errors.append(f"{prefix}.{identity}为未知ID")
                continue
            if row_id in seen:
                errors.append(f"{prefix}.{identity}重复：{row_id}")
                continue
            seen.add(row_id)
            indexes[section][row_id] = copy.deepcopy(row)
            reason = row.get("reason")
            if not isinstance(reason, str):
                errors.append(f"{prefix}.reason须为字符串")
            if section == "event_reviews":
                if not in_enum(row.get("decision"), {"accept", "change", "uncertain"}):
                    errors.append(f"{prefix}.decision非法")
                for checked in ("images_reviewed", "cross_group_checked"):
                    if type(row.get(checked)) is not bool:
                        errors.append(f"{prefix}.{checked}必须显式为bool，不能省略或用0/1替代")
                if in_enum(row.get("decision"), {"accept", "change"}) and (row.get("images_reviewed") is not True or row.get("cross_group_checked") is not True):
                    errors.append(f"{prefix}.accept/change须明确已查看原图并跨组检查，两项均为true")
                if not nonempty(reason):
                    errors.append(f"{prefix}每项事件判断均需一句非空依据")
                proposed, merges = row.get("proposed_event_id"), row.get("merge_with")
                if not isinstance(proposed, str):
                    errors.append(f"{prefix}.proposed_event_id须为字符串，未提出用空字符串")
                valid_merges = isinstance(merges, list) and all(isinstance(item, str) for item in merges)
                if not valid_merges or (valid_merges and (len(merges) != len(set(merges)) or any(item not in expected or item == row_id for item in merges))):
                    errors.append(f"{prefix}.merge_with须为不重复的已知其它claim IDs")
                if row.get("decision") == "change" and not (nonempty(proposed) or (valid_merges and merges)):
                    errors.append(f"{prefix}.change需明确proposed_event_id或merge_with")
                if row.get("decision") == "accept" and (nonempty(proposed) or (valid_merges and merges)):
                    errors.append(f"{prefix}.accept不能同时提出改组/合并，应选change")
            elif section == "link_reviews":
                if not in_enum(row.get("relation"), {"same_event", "same_asset_only", "topic_only", "different", "uncertain"}):
                    errors.append(f"{prefix}.relation非法")
                if not nonempty(reason):
                    errors.append(f"{prefix}每项关联判断均需非空依据")
            else:
                status, label, class_label = row.get("status"), row.get("label"), row.get("class_label")
                refs = row.get("evidence_refs")
                valid_refs = isinstance(refs, list) and all(nonempty(item) for item in refs)
                if not valid_refs or (valid_refs and len(refs) != len(set(refs))):
                    errors.append(f"{prefix}.evidence_refs须为不重复的非空字符串数组")
                if not in_enum(status, {"done", "pending_materials", "unable_to_verify"}):
                    errors.append(f"{prefix}.status非法")
                ready = manifest["task_materials_ready_by_id"][row_id]
                kind = manifest["task_kind_by_id"][row_id]
                if not ready and status != "pending_materials":
                    errors.append(f"{prefix}材料尚未齐备，只能pending_materials，不能靠填表完成核验")
                if not ready and valid_refs and refs:
                    errors.append(f"{prefix}材料尚未齐备时evidence_refs必须为空，不能手改引用冒充固定材料")
                if status == "done":
                    if kind == "temperature_calibration":
                        if label is not None or not isinstance(class_label, str) or class_label not in CLASS_LABELS:
                            errors.append(f"{prefix}校准done须label=null且class_label为四类之一")
                    elif type(label) is not int or label not in (0, 1) or class_label is not None:
                        errors.append(f"{prefix}二元监督done须label严格整数0/1且class_label=null")
                elif label is not None or class_label is not None:
                    errors.append(f"{prefix}未完成/无法核实须label和class_label均为null，不转为0")
                if in_enum(status, {"done", "unable_to_verify"}) and (not nonempty(reason) or not valid_refs or not refs):
                    errors.append(f"{prefix}完成或实际无法核验须有非空理由与证据引用")
    if errors:
        raise ValidationError(errors)
    coverage = {}
    normalized = []
    for section, (identity, expected_key) in SECTIONS.items():
        present = indexes[section]
        expected_ids = manifest[expected_key]
        count = len(expected_ids)
        coverage[section] = {"expected": count, "submitted": len(present), "missing": count-len(present),
                             "submitted_fraction": len(present)/count if count else None,
                             "missing_ids": [value for value in expected_ids if value not in present]}
        if section == "annotation_reviews":
            counts = Counter(row["status"] for row in present.values())
            coverage[section].update(done=counts["done"], pending_materials=counts["pending_materials"],
                                     unable_to_verify=counts["unable_to_verify"],
                                     materials_ready=sum(manifest["task_materials_ready_by_id"].values()),
                                     done_fraction=counts["done"]/count if count else None)
        elif section == "event_reviews":
            coverage[section].update(images_reviewed=sum(row["images_reviewed"] for row in present.values()),
                                     cross_group_checked=sum(row["cross_group_checked"] for row in present.values()))
        for row in present.values():
            normalized.append({"schema_version": 1, "package_id": data["package_id"], "reviewer_slot": data["reviewer_slot"],
                "reviewer_name": data["reviewer_name"], "review_type": section, **row,
                "verification_level": "STRUCTURE_ONLY_SINGLE_HUMAN_SUBMISSION", "human_event_review_complete": False, "TRAINING_READY": False})
    return {"schema_version": 1, "package_id": data["package_id"], "reviewer_slot": data["reviewer_slot"],
            "reviewer_name": data["reviewer_name"], "status": "STRUCTURE_VALID_NOT_TRAINING_READY", "coverage": coverage,
            "human_event_review_complete": False, "TRAINING_READY": False,
            "limitations": ["字段检查不证明审核者实际独立身份或引用真实充分。", "全量候选组填写也不能证明跨组漏合并已解决。", "仍需真人仲裁、事件隔离重划分和真实证据/数值特征验证。", "校准class_label仅供人工辅助复核；正式校准仍须开发原标签及真实logits，本工具不导出校准训练数据。"],
            "validated_single_reviews": normalized}


def compare_returns(a, b, manifest_a, manifest_b):
    first, second = validate_return(a, manifest_a), validate_return(b, manifest_b)
    if first["reviewer_slot"] != "A" or second["reviewer_slot"] != "B":
        raise ValidationError("compare的--a必须对应A包，--b必须对应B包")
    if first["package_id"] != second["package_id"]:
        raise ValidationError("两个返回不属于同一package_id")
    if first["reviewer_name"] == second["reviewer_name"]:
        raise ValidationError("两名审核者reviewer_name精确相同，拒绝作为双人独立审核")
    for key in ("expected_claim_ids", "expected_link_ids", "expected_task_ids"):
        if set(manifest_a[key]) != set(manifest_b[key]):
            raise ValidationError(f"两个包的{key}不一致")
    for key in ("task_kind_by_id", "task_materials_ready_by_id"):
        if manifest_a[key] != manifest_b[key]:
            raise ValidationError(f"两个包的{key}不一致")
    conflicts, incomplete, matching = [], [], Counter()
    for section, (identity, expected_key) in SECTIONS.items():
        left, right = {row[identity]: row for row in a[section]}, {row[identity]: row for row in b[section]}
        for row_id in manifest_a[expected_key]:
            x, y = left.get(row_id), right.get(row_id)
            item = {"section": section, "id": row_id, "A": x, "B": y, "requires_human_action": True}
            if x is None or y is None:
                incomplete.append(dict(item, type="missing_both" if x is None and y is None else ("missing_A" if x is None else "missing_B")))
                continue
            if section == "event_reviews":
                vx = (x["decision"], x["proposed_event_id"], sorted(x["merge_with"]))
                vy = (y["decision"], y["proposed_event_id"], sorted(y["merge_with"]))
                if vx != vy:
                    conflicts.append(dict(item, type="event_judgment_disagreement"))
                elif x["decision"] == "accept":
                    matching[section] += 1
                else:
                    incomplete.append(dict(item, type="matching_change_proposal_needs_human_review" if x["decision"] == "change" else "uncertain_needs_human_review"))
            elif section == "link_reviews":
                if x["relation"] != y["relation"]:
                    conflicts.append(dict(item, type="link_judgment_disagreement"))
                elif x["relation"] == "uncertain":
                    incomplete.append(dict(item, type="uncertain_needs_human_review"))
                else:
                    matching[section] += 1
            elif x["status"] != y["status"]:
                conflicts.append(dict(item, type="annotation_status_disagreement"))
            elif x["status"] != "done":
                incomplete.append(dict(item, type="pending_materials" if x["status"] == "pending_materials" else "unable_to_verify_needs_human_review"))
            elif set(x["evidence_refs"]) != set(y["evidence_refs"]):
                conflicts.append(dict(item, type="annotation_evidence_scope_mismatch"))
            elif (x["label"], x["class_label"]) != (y["label"], y["class_label"]):
                conflicts.append(dict(item, type="annotation_label_disagreement"))
            else:
                matching[section] += 1
    return {"schema_version": 1, "package_id": a["package_id"], "status": "COMPARED_WITHOUT_ADJUDICATION",
            "reviewers": {"A": a["reviewer_name"], "B": b["reviewer_name"]},
            "coverage": {"A": first["coverage"], "B": second["coverage"]},
            "matching_submitted_judgments": dict(matching), "conflict_count": len(conflicts), "incomplete_count": len(incomplete),
            "conflicts": conflicts, "incomplete": incomplete, "human_event_review_complete": False, "TRAINING_READY": False,
            "limitations": first["limitations"],
            "validated_single_reviews": {"A": first["validated_single_reviews"], "B": second["validated_single_reviews"]}}


def write_outputs(output_dir, result):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValidationError("输出目录已有内容，拒绝覆盖返回或既有核查结果")
    output.mkdir(parents=True, exist_ok=True)
    def dump(name, value):
        with (output/name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    def lines(name, rows):
        with (output/name).open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    summary = {key: value for key, value in result.items() if key not in {"validated_single_reviews", "conflicts", "incomplete"}}
    if result["status"] == "COMPARED_WITHOUT_ADJUDICATION":
        dump("comparison_report.json", summary)
        dump("conflicts.json", result["conflicts"])
        dump("incomplete.json", result["incomplete"])
        for slot in ("A", "B"):
            lines(f"validated_single_review_{slot}.jsonl", result["validated_single_reviews"][slot])
    else:
        dump("validation_report.json", summary)
        lines("validated_single_review.jsonl", result["validated_single_reviews"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    single = sub.add_parser("validate")
    single.add_argument("--file", type=Path, required=True)
    single.add_argument("--manifest", type=Path, required=True)
    single.add_argument("--out", type=Path)
    compare = sub.add_parser("compare")
    for argument in ("a", "b", "manifest-a", "manifest-b", "out"):
        compare.add_argument("--"+argument, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            result = validate_return(read_json(args.file), read_json(args.manifest))
            output = args.out or args.file.with_name(args.file.stem+".validation")
        else:
            result = compare_returns(read_json(args.a), read_json(args.b), read_json(args.manifest_a), read_json(args.manifest_b))
            output = args.out
        summary = write_outputs(output, result)
        print(json.dumps({"output": str(output.resolve()), **summary}, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValidationError, OSError) as exc:
        print(json.dumps({"status": "REJECTED", "errors": exc.errors if isinstance(exc, ValidationError) else [str(exc)],
                          "human_event_review_complete": False, "TRAINING_READY": False}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
