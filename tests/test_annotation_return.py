"""双人独立审核返回的结构、未完成语义和分歧隔离契约。"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("annotation_return_tool", Path(__file__).resolve().parents[1]/"tools/validate_annotation_return.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def manifest(slot="A", ready=False):
    kinds = {"T-source": "source", "T-quality": "quality", "T-sufficiency": "sufficiency", "T-calibration": "temperature_calibration"}
    return {"schema_version": 1, "package_id": "annotation-20261117-v1", "reviewer_slot": slot,
            "expected_claim_ids": ["C000001", "C000002"], "expected_link_ids": ["L000001"],
            "expected_task_ids": list(kinds), "task_kind_by_id": kinds,
            "task_materials_ready_by_id": {task_id: ready for task_id in kinds}}


def returned(slot="A", name="审核人甲"):
    return {"schema_version": 1, "package_id": "annotation-20261117-v1", "reviewer_slot": slot,
            "reviewer_name": name, "event_reviews": [], "link_reviews": [], "annotation_reviews": []}


def event(claim_id="C000001", **changes):
    return {"claim_id": claim_id, "decision": "accept", "proposed_event_id": "", "merge_with": [],
            "reason": "已查看原图并比对相关公开声明。", "images_reviewed": True, "cross_group_checked": True, **changes}


def annotation(task_id="T-source", **changes):
    return {"task_id": task_id, "status": "pending_materials", "label": None, "class_label": None,
            "reason": "未收到已归档证据/模型历史", "evidence_refs": [], **changes}


def test_blank_and_material_pending_never_become_negative_or_ready():
    data = returned()
    data["annotation_reviews"] = [annotation(task_id) for task_id in manifest()["expected_task_ids"]]
    result = tool.validate_return(data, manifest())
    coverage = result["coverage"]["annotation_reviews"]
    assert coverage["submitted"] == 4 and coverage["done"] == 0 and coverage["pending_materials"] == 4
    assert all(row["label"] is None and row["class_label"] is None for row in result["validated_single_reviews"])
    assert result["human_event_review_complete"] is False and result["TRAINING_READY"] is False


def test_partial_submissions_count_missing_without_fabricating_rows():
    data = returned()
    data["event_reviews"] = [event()]
    result = tool.validate_return(data, manifest())
    assert result["coverage"]["event_reviews"]["missing_ids"] == ["C000002"]
    assert len(result["validated_single_reviews"]) == 1
    assert result["coverage"]["annotation_reviews"]["done"] == 0


@pytest.mark.parametrize("section, row", [("event_reviews", event("C999999")),
    ("link_reviews", {"link_id": "L999999", "relation": "same_event", "reason": "依据"}),
    ("annotation_reviews", annotation("T-unknown"))])
def test_unknown_ids_rejected(section, row):
    data = returned()
    data[section] = [row]
    with pytest.raises(tool.ValidationError, match="未知ID"):
        tool.validate_return(data, manifest())


@pytest.mark.parametrize("section, row", [("event_reviews", event()),
    ("link_reviews", {"link_id": "L000001", "relation": "topic_only", "reason": "仅同一主题。"}),
    ("annotation_reviews", annotation())])
def test_duplicate_ids_rejected(section, row):
    data = returned()
    data[section] = [row, copy.deepcopy(row)]
    with pytest.raises(tool.ValidationError, match="重复"):
        tool.validate_return(data, manifest())


@pytest.mark.parametrize("field, value", [("package_id", "another-package"), ("reviewer_slot", "B"),
    ("schema_version", True), ("reviewer_name", "  "), ("reviewer_name", " 甲 ")])
def test_package_slot_version_and_name_checked(field, value):
    data = returned()
    data[field] = value
    with pytest.raises(tool.ValidationError):
        tool.validate_return(data, manifest())


@pytest.mark.parametrize("changes", [{"images_reviewed": False}, {"cross_group_checked": False},
    {"images_reviewed": 1}, {"reason": ""}, {"decision": "automatic_accept"}, {"decision": []},
    {"decision": "change", "proposed_event_id": "", "merge_with": []},
    {"decision": "change", "merge_with": ["C000001"]},
    {"decision": "change", "merge_with": ["C999999"]},
    {"decision": "change", "merge_with": ["C000002", "C000002"]}])
def test_event_decisions_require_explicit_checks_reasons_and_valid_change(changes):
    data = returned()
    data["event_reviews"] = [event(**changes)]
    with pytest.raises(tool.ValidationError):
        tool.validate_return(data, manifest())


def test_uncertain_event_may_leave_checks_false_but_still_requires_reason():
    data = returned()
    data["event_reviews"] = [event(decision="uncertain", images_reviewed=False, cross_group_checked=False, reason="图片信息不足，尚不能判断。")]
    result = tool.validate_return(data, manifest())
    assert result["coverage"]["event_reviews"]["images_reviewed"] == 0


@pytest.mark.parametrize("relation", ["same_event", "same_asset_only", "topic_only", "different", "uncertain"])
def test_all_five_link_relations_accepted_with_reason(relation):
    data = returned()
    data["link_reviews"] = [{"link_id": "L000001", "relation": relation, "reason": "已对照两条公开声明。"}]
    assert tool.validate_return(data, manifest())["coverage"]["link_reviews"]["submitted"] == 1


@pytest.mark.parametrize("status", ["done", "unable_to_verify"])
def test_materials_absent_blocks_claimed_completion_or_unverifiability(status):
    data = returned()
    data["annotation_reviews"] = [annotation(status=status, label=1 if status == "done" else None,
        reason="人工声称已经核验", evidence_refs=["claimed-ref"])]
    with pytest.raises(tool.ValidationError, match="材料尚未齐备"):
        tool.validate_return(data, manifest())


def test_materials_absent_pending_cannot_smuggle_in_unverified_references():
    data = returned()
    data["annotation_reviews"] = [annotation(evidence_refs=["manually-added-unfixed-reference"])]
    with pytest.raises(tool.ValidationError, match="evidence_refs必须为空"):
        tool.validate_return(data, manifest())


@pytest.mark.parametrize("label", [True, False, 0.0, 2, "1", None])
def test_binary_done_rejects_bool_and_non_integer_labels(label):
    data = returned()
    data["annotation_reviews"] = [annotation(status="done", label=label, reason="核验依据", evidence_refs=["E1", "E2"])]
    with pytest.raises(tool.ValidationError, match="严格整数0/1"):
        tool.validate_return(data, manifest(ready=True))


@pytest.mark.parametrize("label", [0, 1])
def test_material_ready_binary_records_remain_single_review_only(label):
    data = returned()
    data["annotation_reviews"] = [annotation(status="done", label=label, reason="已核对真实材料。", evidence_refs=["E1", "E2"])]
    result = tool.validate_return(data, manifest(ready=True))
    assert result["coverage"]["annotation_reviews"]["done"] == 1
    assert result["TRAINING_READY"] is False


@pytest.mark.parametrize("class_label", sorted(tool.CLASS_LABELS))
def test_calibration_accepts_four_class_labels_only_as_auxiliary_review(class_label):
    data = returned()
    data["annotation_reviews"] = [annotation("T-calibration", status="done", class_label=class_label,
        reason="辅助人工核对。", evidence_refs=["model-history-1"])]
    result = tool.validate_return(data, manifest(ready=True))
    assert result["validated_single_reviews"][0]["label"] is None
    assert result["TRAINING_READY"] is False
    assert any("不导出校准训练" in text for text in result["limitations"])


@pytest.mark.parametrize("changes", [{"label": 1, "class_label": "Supported"}, {"class_label": "true"},
    {"class_label": 1}, {"class_label": None}, {"class_label": []}])
def test_calibration_rejects_binary_or_invalid_class_labels(changes):
    data = returned()
    data["annotation_reviews"] = [annotation("T-calibration", status="done", reason="依据", evidence_refs=["H1"], **changes)]
    with pytest.raises(tool.ValidationError, match="校准done"):
        tool.validate_return(data, manifest(ready=True))


@pytest.mark.parametrize("changes", [{"reason": ""}, {"evidence_refs": []}, {"evidence_refs": [""]},
    {"evidence_refs": ["E1", "E1"]}, {"evidence_refs": [{"id": "E1"}]}])
def test_done_requires_reason_and_nonempty_reference_strings(changes):
    data = returned()
    row = annotation(status="done", label=1, reason="依据", evidence_refs=["E1"])
    row.update(changes)
    data["annotation_reviews"] = [row]
    with pytest.raises(tool.ValidationError):
        tool.validate_return(data, manifest(ready=True))


def test_pending_and_unable_labels_cannot_be_zero_filled():
    for status in ("pending_materials", "unable_to_verify"):
        data = returned()
        data["annotation_reviews"] = [annotation(status=status, label=0, evidence_refs=["E1"], reason="依据")]
        with pytest.raises(tool.ValidationError, match="均为null"):
            tool.validate_return(data, manifest(ready=True))


def test_same_exact_reviewer_name_cannot_be_two_independent_people():
    with pytest.raises(tool.ValidationError, match="精确相同"):
        tool.compare_returns(returned(), returned("B"), manifest(), manifest("B"))


def test_compare_missing_and_pending_are_incomplete_not_zero_or_consensus():
    a, b = returned(), returned("B", "审核人乙")
    a["annotation_reviews"] = [annotation()]
    b["annotation_reviews"] = [annotation()]
    result = tool.compare_returns(a, b, manifest(), manifest("B"))
    assert result["conflict_count"] == 0
    assert any(item["type"] == "pending_materials" for item in result["incomplete"])
    assert any(item["type"] == "missing_both" for item in result["incomplete"])
    assert result["TRAINING_READY"] is False and result["human_event_review_complete"] is False


def test_matching_accepts_still_cannot_certify_full_event_review():
    a, b = returned(), returned("B", "审核人乙")
    a["event_reviews"] = b["event_reviews"] = [event("C000001"), event("C000002")]
    result = tool.compare_returns(a, b, manifest(), manifest("B"))
    assert result["matching_submitted_judgments"]["event_reviews"] == 2
    assert result["human_event_review_complete"] is False
    assert any("跨组漏合并" in text for text in result["limitations"])


def test_same_label_on_different_evidence_is_scope_conflict():
    a, b = returned(), returned("B", "审核人乙")
    a["annotation_reviews"] = [annotation(status="done", label=1, reason="依据甲", evidence_refs=["E1", "E2"])]
    b["annotation_reviews"] = [annotation(status="done", label=1, reason="依据乙", evidence_refs=["E3", "E4"])]
    result = tool.compare_returns(a, b, manifest(ready=True), manifest("B", ready=True))
    assert result["conflicts"][0]["type"] == "annotation_evidence_scope_mismatch"


def test_compare_reports_event_link_and_label_disagreements_without_adjudication():
    a, b = returned(), returned("B", "审核人乙")
    a["event_reviews"], b["event_reviews"] = [event()], [event(decision="change", merge_with=["C000002"])]
    a["link_reviews"] = [{"link_id": "L000001", "relation": "same_asset_only", "reason": "图像相同"}]
    b["link_reviews"] = [{"link_id": "L000001", "relation": "same_event", "reason": "事件相同"}]
    a["annotation_reviews"] = [annotation(status="done", label=0, reason="依据甲", evidence_refs=["E1"])]
    b["annotation_reviews"] = [annotation(status="done", label=1, reason="依据乙", evidence_refs=["E1"])]
    result = tool.compare_returns(a, b, manifest(ready=True), manifest("B", ready=True))
    assert {row["type"] for row in result["conflicts"]} == {"event_judgment_disagreement", "link_judgment_disagreement", "annotation_label_disagreement"}
    assert "adjudicated_label" not in json.dumps(result)


def test_output_is_single_review_jsonl_and_refuses_overwrite(tmp_path):
    data = returned()
    data["event_reviews"] = [event()]
    result = tool.validate_return(data, manifest())
    tool.write_outputs(tmp_path/"validated", result)
    row = json.loads((tmp_path/"validated/validated_single_review.jsonl").read_text(encoding="utf-8"))
    assert row["reviewer_name"] == "审核人甲" and row["TRAINING_READY"] is False
    with pytest.raises(tool.ValidationError, match="拒绝覆盖"):
        tool.write_outputs(tmp_path/"validated", result)


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_json_duplicate_keys_and_nonfinite_values_rejected(tmp_path, payload):
    path = tmp_path/"bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(tool.ValidationError):
        tool.read_json(path)


def test_cli_validation_creates_report_and_does_not_create_training_files(tmp_path):
    input_path, manifest_path = tmp_path/"returned.json", tmp_path/"manifest.json"
    input_path.write_text(json.dumps(returned()), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
    assert tool.main(["validate", "--file", str(input_path), "--manifest", str(manifest_path)]) == 0
    report = json.loads((tmp_path/"returned.validation/validation_report.json").read_text(encoding="utf-8"))
    assert report["TRAINING_READY"] is False
    assert sorted(path.name for path in (tmp_path/"returned.validation").iterdir()) == ["validated_single_review.jsonl", "validation_report.json"]
