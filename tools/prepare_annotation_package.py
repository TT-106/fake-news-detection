"""生成待人工取证/复核的中性材料；不生成任何人工结论或可训练标签。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"questions", "answers", "article", "justification", "metadata", "gold_label", "dataset_label"}
ROLES = ("module_fit", "temperature_calibration", "mechanism_validation")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def table(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json.dumps(row[name], ensure_ascii=False) if isinstance(row.get(name), (list, dict)) else row.get(name) for name in columns})


def public_claim(sample, image_dir, root):
    # 严格白名单：绝不把 label、QA、核查文章、事后解释或 metadata 传给材料模板。
    images = []
    for raw_name in sample.get("claim_images") or []:
        name = str(raw_name).replace("\\", "/").rsplit("/", 1)[-1]
        path = (image_dir / name).resolve()
        if path.parent != image_dir.resolve() or not path.is_file():
            raise ValueError(f"声明原图缺失或越界：{name}")
        images.append({"file": path.relative_to(root.resolve()).as_posix(), "sha256": sha(path)})
    return {"claim_text": sample.get("claim_text") or "", "claim_date": sample.get("date") or "",
            "claim_location": sample.get("location") or "", "claim_images": images}


def evidence_slot():
    return {"evidence_id": None, "url": None, "snapshot_id": None, "snapshot_sha256": None,
            "start": None, "end": None, "text": None, "retrieved_at": None,
            "published_at": None, "author": None}


def task(kind, role, neutral, event, claim):
    row = {"task_id": f"{role}-{kind}-{neutral}", "claim_id": neutral, "event_group": event,
           "role": role, "task_kind": kind, **claim, "record_status": "PENDING_EVIDENCE_AND_INDEPENDENT_HUMAN_REVIEW",
           "training_ready": False, "reviewer": None, "label": None, "rationale": None,
           "features": None, "evidence_proofs": None}
    if kind == "source":
        row.update(evidence_a=evidence_slot(), evidence_b=evidence_slot(), relation_decision=None)
    elif kind == "quality":
        row.update(atom_id=None, atom_text=None, evidence=evidence_slot())
    elif kind == "sufficiency":
        row.update(history_id=None, atomic_claims=None, terminal_prediction=None, cited_evidence=None)
    else:
        row.update(frozen_model_id=None, logits=None, reviewed_class_label=None)
    return row


def contains_forbidden(value):
    if isinstance(value, dict):
        return any(key in FORBIDDEN or contains_forbidden(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden(item) for item in value)
    return False


def validate(package_dir, root=ROOT):
    package_dir, root = Path(package_dir), Path(root)
    manifest = read(package_dir / "manifest.json")
    key = read(package_dir / "private" / "claim_key.json")
    neutral_to_role = {row["claim_id"]: row["role"] for row in key}
    expected = manifest["source_role_counts"]
    task_counts, reviewers = {}, []
    for reviewer in ("reviewer_a", "reviewer_b"):
        path = package_dir / "annotation_review" / reviewer / "tasks.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        reviewers.append(rows)
        counts = Counter((row["role"], row["task_kind"]) for row in rows)
        assert counts == Counter({("module_fit", "source"): expected["module_fit"],
                                  ("module_fit", "quality"): expected["module_fit"],
                                  ("module_fit", "sufficiency"): expected["module_fit"],
                                  ("temperature_calibration", "temperature_calibration"): expected["temperature_calibration"],
                                  ("mechanism_validation", "source"): expected["mechanism_validation"]})
        assert len({row["task_id"] for row in rows}) == len(rows)
        for row in rows:
            assert row["role"] in ROLES and neutral_to_role[row["claim_id"]] == row["role"]
            assert row["reviewer"] is None and row["label"] is None and row["features"] is None
            assert row["evidence_proofs"] is None and row["rationale"] is None and row["training_ready"] is False
            assert not contains_forbidden(row)
            for image in row["claim_images"]:
                assert (root / image["file"]).is_file() and sha(root / image["file"]) == image["sha256"]
        task_counts[reviewer] = len(rows)
    assert reviewers[0] == reviewers[1], "独立空副本应包含相同任务，但不包含对方实际结论"
    for reviewer in ("reviewer_a", "reviewer_b"):
        event_rows = [json.loads(line) for line in (package_dir / "event_review" / reviewer / "claims.jsonl").read_text(encoding="utf-8").splitlines() if line]
        assert len(event_rows) == manifest["all_claim_count"]
        assert len({row["claim_id"] for row in event_rows}) == len(event_rows)
        assert all(row["accept_candidate_group"] is None and row["reviewer"] is None and row["proposed_event_id"] is None for row in event_rows)
        assert all(not contains_forbidden(row) for row in event_rows)
    for file, expected_sha in manifest["file_sha256"].items():
        assert sha(package_dir / file) == expected_sha, f"材料已修改：{file}"
    assert manifest["human_event_review_complete"] is False and manifest["training_ready"] is False
    sentinel = {"claim_text": "SAFE PUBLIC CLAIM", "claim_images": [], "label": "SECRET_GOLD",
                "questions": [{"answers": "SECRET_QA"}], "article": "SECRET_ARTICLE", "metadata": {"hidden": "SECRET_METADATA"}}
    assert "SECRET" not in json.dumps(public_claim(sentinel, root / "data/averimatec/images", root))
    return {"status": "VALID_PENDING_PACKAGE_NOT_TRAINING_READY", "task_counts": task_counts,
            "all_claim_count": manifest["all_claim_count"], "event_group_count": manifest["event_group_count"],
            "source_role_counts": expected, "all_human_labels_reviewers_features_null": True,
            "final_eval_excluded_from_annotation_tasks": True, "qa_gold_article_metadata_not_exported": True,
            "claim_image_refs_and_sha_verified": True, "human_event_review_complete": False,
            "training_ready": False, "file_checksums_verified": True}


def generate(package_id="review-20261117-01", root=ROOT):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", package_id):
        raise ValueError("package-id 只能含英文字母、数字、点、短横线、下划线")
    pending = root / "data" / "annotations" / "pending"
    output = (pending / package_id).resolve()
    output.relative_to(pending.resolve())
    if output.exists():
        raise ValueError("材料目录已存在，拒绝覆盖人工填写；请用 --verify-only 检查或另选 package-id")
    split_path = root / "outputs/method_repair/provisional_split.json"
    split = read(split_path)
    samples = {}
    for partition in ("train", "val"):
        path = root / "data/averimatec" / (partition + ".json")
        if sha(path) != split["dataset_sha256"][partition + ".json"]:
            raise ValueError("原始数据版本与候选划分不一致，先重新审核划分")
        samples.update({f"{partition}:{i}": sample for i, sample in enumerate(read(path))})
    roles = {cid: role for role, ids in split["split"].items() for cid in ids}
    roles.update({cid: "quarantined" for cid in split.get("quarantined_train_claims", [])})
    if set(roles) != set(samples) or set(split["event_groups"]) != set(samples):
        raise ValueError("候选划分或事件组未完整覆盖数据")
    if set(split["split"]["final_eval"]) != {cid for cid in samples if cid.startswith("val:")}:
        raise ValueError("最终评测必须完整保留官方 val，仅供事件审核，不生成监督任务")
    if {role: len(split["split"][role]) for role in ROLES} != {"module_fit": 45, "temperature_calibration": 15, "mechanism_validation": 60}:
        raise ValueError("本版材料固定45/15/60模板，划分变化后须先复核材料协议")
    neutral = {cid: f"C{i:06d}" for i, cid in enumerate(samples, 1)}
    group_names = {gid: f"G{i:06d}" for i, gid in enumerate(dict.fromkeys(split["event_groups"].values()), 1)}
    claims = {cid: public_claim(sample, root / "data/averimatec/images", root) for cid, sample in samples.items()}
    key = [{"claim_id": neutral[cid], "original_id": cid, "role": roles[cid], "event_group": group_names[split["event_groups"][cid]],
            "original_candidate_group": split["event_groups"][cid]} for cid in samples]
    annotation_tasks = []
    for role in ROLES:
        kinds = ("source", "quality", "sufficiency") if role == "module_fit" else (("source",) if role == "mechanism_validation" else ("temperature_calibration",))
        for cid in split["split"][role]:
            for kind in kinds:
                annotation_tasks.append(task(kind, role, neutral[cid], group_names[split["event_groups"][cid]], claims[cid]))
    event_rows = [{"claim_id": neutral[cid], "provisional_event_id": group_names[split["event_groups"][cid]],
                   "role": roles[cid], "dataset_partition": cid.split(":")[0], **claims[cid],
                   "usage_boundary": "FINAL_EVAL_EVENT_REVIEW_ONLY_NO_TRAINING" if roles[cid] == "final_eval" else roles[cid],
                   "accept_candidate_group": None, "proposed_event_id": None, "suggest_merge_with": None,
                   "reviewer": None, "review_note": None} for cid in samples]
    grouped = defaultdict(list)
    for cid, gid in split["event_groups"].items():
        grouped[gid].append(cid)
    pairs = []
    for gid, cids in grouped.items():
        for a, b in itertools.combinations(cids, 2):
            a_text = " ".join(claims[a]["claim_text"].casefold().split())
            b_text = " ".join(claims[b]["claim_text"].casefold().split())
            shared_images = sorted({i["sha256"] for i in claims[a]["claim_images"]} & {i["sha256"] for i in claims[b]["claim_images"]})
            reasons = (["same_normalized_public_claim_text"] if a_text and a_text == b_text else []) + (["same_claim_image_sha256"] if shared_images else [])
            if not reasons:
                reasons = ["candidate_group_from_existing_audit_requires_independent_confirmation"]
            pairs.append({"event_group": group_names[gid], "claim_a": neutral[a], "claim_b": neutral[b],
                          "role_a": roles[a], "role_b": roles[b], "candidate_link_basis": reasons,
                          "shared_claim_image_sha256": shared_images, "accept_link": None, "reviewer": None, "review_note": None})
    output.mkdir(parents=True, exist_ok=False)
    write(output / "private" / "claim_key.json", key)
    write(output / "private" / "split_review_state.json", {"source_split_sha256": sha(split_path),
        "dataset_sha256": split["dataset_sha256"], "human_event_review_complete": False,
        "training_ready": False, "approved_event_groups": None, "approved_split": None,
        "reviewer_a": None, "reviewer_b": None, "adjudicator": None})
    for reviewer in ("reviewer_a", "reviewer_b"):
        jsonl(output / "annotation_review" / reviewer / "tasks.jsonl", annotation_tasks)
        table(output / "annotation_review" / reviewer / "tasks.csv", annotation_tasks,
              ["task_id", "claim_id", "event_group", "role", "task_kind", "claim_text", "claim_date", "claim_location", "claim_images", "reviewer", "label", "rationale", "features", "evidence_proofs"])
        jsonl(output / "event_review" / reviewer / "claims.jsonl", event_rows)
        table(output / "event_review" / reviewer / "claims.csv", event_rows,
              ["claim_id", "provisional_event_id", "role", "dataset_partition", "claim_text", "claim_date", "claim_location", "claim_images", "usage_boundary", "accept_candidate_group", "proposed_event_id", "suggest_merge_with", "reviewer", "review_note"])
        jsonl(output / "event_review" / reviewer / "candidate_links.jsonl", pairs)
        table(output / "event_review" / reviewer / "candidate_links.csv", pairs,
              ["event_group", "claim_a", "claim_b", "role_a", "role_b", "candidate_link_basis", "shared_claim_image_sha256", "accept_link", "reviewer", "review_note"])
    jsonl(output / "adjudication" / "annotation_tasks.jsonl", [{"task_id": row["task_id"], "reviewer_a": None, "label_a": None,
        "reviewer_b": None, "label_b": None, "adjudicator": None, "adjudicated_label": None, "rationale": None} for row in annotation_tasks])
    jsonl(output / "adjudication" / "event_groups.jsonl", [{"provisional_event_id": group_names[gid], "reviewer_a": None,
        "decision_a": None, "reviewer_b": None, "decision_b": None, "adjudicator": None, "approved_group": None,
        "rationale": None} for gid in grouped])
    readme = """# 待人工审核包\n\n状态：未取证、未标注、未复核，不能用于训练或正式评测。\n\n1. `event_review/reviewer_a` 与 `reviewer_b` 为独立事件复核副本。全量声明含最终评测的公开图文，只能用于事件隔离审核；不能作为训练或模型/检索提示。候选同组不是已证实同事件；还需查找跨组的遗漏关联。接受/否决、合并建议及备注均为空。\n2. `annotation_review/reviewer_a` 与 `reviewer_b` 为独立监督任务模板：模块拟合45条各含来源/质量/充分性三类任务，校准15条，来源机制验证60条。每条仅是待选择证据/历史的工作单，不是一个已完成监督样本；同一声明可能需要多对片段/多条原子声明/多个历史。\n3. 在取得真实归档证据前，不得填写或猜测 features/label。二人独立填写，互不读取对方结论；分歧由第三人处理。`adjudication` 保持空白，不能由脚本双签。\n4. `private` 仅给协调者保管，保存中性ID到原数据索引的映射。不得把原始数据的真值、QA、核查链接或解释抄入模型/检索输入。原图file相对项目根目录；本包不重复复制图片。\n5. `tasks.csv`便于分派与填写审核信息；完整证据、历史及模型量须在`tasks.jsonl`对应槽位整理。CSV的空格表示未填，JSON的null表示未有结论，不可解释为负例或无法核实的有效标注。\n6. 本包不导出训练接口文件，不设置human_event_review_complete=true。完成独立审核、仲裁与事件重划分后，再人工导出规范的三类训练JSONL/独立校准/机制评测材料。\n\n详细字段与步骤见项目 `docs/ANNOTATION_WORKFLOW.md`。\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    checksums = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}
    write(output / "manifest.json", {"package_id": package_id, "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "status": "PENDING_EVIDENCE_AND_INDEPENDENT_HUMAN_REVIEW", "training_ready": False,
        "human_event_review_complete": False, "source_split_path": split_path.relative_to(root).as_posix(),
        "source_split_sha256": sha(split_path), "dataset_sha256": split["dataset_sha256"],
        "source_role_counts": {role: len(split["split"][role]) for role in ROLES},
        "all_claim_count": len(samples), "event_group_count": len(grouped), "candidate_link_count": len(pairs),
        "task_templates_per_reviewer": len(annotation_tasks), "task_template_note": "不是已完成标注行；未取得真实证据/历史，features和人工结论全null。",
        "file_sha256": checksums})
    report = validate(output, root)
    write(output / "validation_report.json", report)
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-id", default="review-20261117-01")
    parser.add_argument("--verify-only", type=Path, help="只验证已有空白包；已填写材料应保留原包并另做人工审核，不能当作空白包验证")
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            report = validate(args.verify_only)
            output = args.verify_only
        else:
            output, report = generate(args.package_id)
        print(json.dumps({"package_path": str(output), **report}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, AssertionError, KeyError) as exc:
        print(f"标注准备材料：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
