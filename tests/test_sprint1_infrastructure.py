"""Sprint 1 基础设施回归测试：事件组级 Cluster Bootstrap 与人工回收转规范 JSONL 验真。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from src.experiment_runner import _load_rows
from src.research_contracts import ProtocolError
from src.research_evaluation import paired_bootstrap
from tools.convert_annotation_return import convert_returns, write_conversion


def test_cluster_bootstrap_matches_claim_bootstrap_for_singletons():
    left = {f"c{i}": float(i % 3) * 0.25 for i in range(12)}
    right = {f"c{i}": 0.1 for i in range(12)}
    groups = {cid: f"g_{cid}" for cid in left}

    res_claim = paired_bootstrap(left, right, seed=7, n_boot=500)
    res_cluster = paired_bootstrap(left, right, seed=7, n_boot=500, event_groups=groups)

    assert res_claim["resampling_unit"] == "claim"
    assert res_cluster["resampling_unit"] == "event_group"
    assert res_cluster["num_clusters"] == 12
    assert res_cluster["max_cluster_size"] == 1
    assert res_cluster["mean_difference"] == pytest.approx(res_claim["mean_difference"])
    assert res_cluster["mean_difference_cluster_equal_weight"] == pytest.approx(res_claim["mean_difference"])


def test_cluster_bootstrap_widens_ci_under_within_cluster_correlation():
    # 4 clusters, each with 5 identical correlated claims (2 clusters = +1, 2 clusters = -1)
    left, right, groups = {}, {}, {}
    for g in range(4):
        val = 1.0 if g < 2 else 0.0
        for k in range(5):
            cid = f"c_{g}_{k}"
            left[cid] = val
            right[cid] = 0.5
            groups[cid] = f"event:{g}"

    claim_ci = paired_bootstrap(left, right, seed=42, n_boot=2000)["95ci"]
    cluster_res = paired_bootstrap(left, right, seed=42, n_boot=2000, event_groups=groups)
    cluster_ci = cluster_res["95ci"]

    assert cluster_res["num_clusters"] == 4
    assert cluster_res["max_cluster_size"] == 5
    # With only 4 clusters of ±0.5, cluster CI must be strictly wider than naive 20-claim CI
    assert (cluster_ci[1] - cluster_ci[0]) > (claim_ci[1] - claim_ci[0])


def test_cluster_bootstrap_rejects_missing_group():
    with pytest.raises(ProtocolError):
        paired_bootstrap({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0}, event_groups={"a": "g1"})


def _make_snapshot(tmp_path: Path, name: str, body: str) -> dict:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8", newline="\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return {
        "evidence_id": f"ev-{name}",
        "url": f"https://example.org/{name}",
        "snapshot_path": str(p),
        "snapshot_sha256": sha,
        "start": 0,
        "end": len(body),
        "text": body,
        "retrieved_at": "2026-11-21T00:00:00+00:00",
        "published_at": "2024-01-01",
        "author": "Reporter",
        "image_provenance_verified": False,
        "provenance_proof": "",
    }


def _fixture_bundle(tmp_path: Path):
    ev1 = _make_snapshot(tmp_path, "doc1.txt", "Original wire dispatch by Reuters.")
    ev2 = _make_snapshot(tmp_path, "doc2.txt", "Reprinted dispatch citing Reuters.")

    manifest_base = {
        "schema_version": 1,
        "package_id": "pkg-test-01",
        "expected_claim_ids": [],
        "expected_link_ids": [],
        "expected_task_ids": ["T_SRC_AGREE", "T_SRC_DISAGREE", "T_SRC_UNKNOWN", "T_QUAL", "T_EVAL", "T_CAL"],
        "task_kind_by_id": {
            "T_SRC_AGREE": "source",
            "T_SRC_DISAGREE": "source",
            "T_SRC_UNKNOWN": "source",
            "T_QUAL": "quality",
            "T_EVAL": "source",
            "T_CAL": "temperature_calibration",
        },
        "task_materials_ready_by_id": {
            "T_SRC_AGREE": True,
            "T_SRC_DISAGREE": True,
            "T_SRC_UNKNOWN": True,
            "T_QUAL": True,
            "T_EVAL": True,
            "T_CAL": True,
        },
    }
    manifest_a = {**manifest_base, "reviewer_slot": "A"}
    manifest_b = {**manifest_base, "reviewer_slot": "B"}

    ret_a = {
        "schema_version": 1, "package_id": "pkg-test-01", "reviewer_slot": "A", "reviewer_name": "Alice",
        "event_reviews": [], "link_reviews": [],
        "annotation_reviews": [
            {"task_id": "T_SRC_AGREE", "status": "done", "label": 1, "class_label": None, "reason": "Same wire text", "evidence_refs": [ev1["evidence_id"], ev2["evidence_id"]]},
            {"task_id": "T_SRC_DISAGREE", "status": "done", "label": 0, "class_label": None, "reason": "Distinct details", "evidence_refs": [ev1["evidence_id"]]},
            {"task_id": "T_SRC_UNKNOWN", "status": "unable_to_verify", "label": None, "class_label": None, "reason": "No clear credit", "evidence_refs": [ev1["evidence_id"]]},
            {"task_id": "T_QUAL", "status": "done", "label": 1, "class_label": None, "reason": "Traceable primary source", "evidence_refs": [ev1["evidence_id"]]},
            {"task_id": "T_EVAL", "status": "done", "label": 1, "class_label": None, "reason": "Reprint match", "evidence_refs": [ev1["evidence_id"], ev2["evidence_id"]]},
            {"task_id": "T_CAL", "status": "done", "label": None, "class_label": "Refuted", "reason": "Refuted by photo", "evidence_refs": [ev1["evidence_id"]]},
        ],
    }
    ret_b = {
        "schema_version": 1, "package_id": "pkg-test-01", "reviewer_slot": "B", "reviewer_name": "Bob",
        "event_reviews": [], "link_reviews": [],
        "annotation_reviews": [
            {"task_id": "T_SRC_AGREE", "status": "done", "label": 1, "class_label": None, "reason": "Exact citation chain", "evidence_refs": [ev1["evidence_id"], ev2["evidence_id"]]},
            {"task_id": "T_SRC_DISAGREE", "status": "done", "label": 1, "class_label": None, "reason": "Same origin", "evidence_refs": [ev1["evidence_id"]]},
            {"task_id": "T_SRC_UNKNOWN", "status": "unable_to_verify", "label": None, "class_label": None, "reason": "Insufficient metadata", "evidence_refs": [ev1["evidence_id"]]},
            {"task_id": "T_QUAL", "status": "done", "label": 1, "class_label": None, "reason": "Complete excerpt", "evidence_refs": [ev1["evidence_id"]]},
            {"task_id": "T_EVAL", "status": "done", "label": 1, "class_label": None, "reason": "Confirmed reprint", "evidence_refs": [ev1["evidence_id"], ev2["evidence_id"]]},
            {"task_id": "T_CAL", "status": "done", "label": None, "class_label": "Refuted", "reason": "Refuted by archive", "evidence_refs": [ev1["evidence_id"]]},
        ],
    }
    task_objects = {
        "tasks": {
            "T_SRC_AGREE": {"claim_id": "train:1", "export_kind": "source", "features": [1.0, 0.9, 0.0, 0.85, 0.0, 1.0], "evidence_proofs": [ev1, ev2]},
            "T_SRC_DISAGREE": {"claim_id": "train:2", "export_kind": "source", "features": [0.0, 0.5, 0.0, 0.4, 0.0, 1.0], "evidence_proofs": [ev1]},
            "T_SRC_UNKNOWN": {"claim_id": "train:3", "export_kind": "source", "features": [0.0, 0.0, 0.0, 0.2, 0.0, 0.0], "evidence_proofs": [ev1]},
            "T_QUAL": {"claim_id": "train:1", "export_kind": "quality", "features": [1.0, 1.0, 0.9], "evidence_proofs": [ev1]},
            "T_EVAL": {"claim_id": "train:10", "export_kind": "source_eval", "evidence_a": ev1["evidence_id"], "evidence_b": ev2["evidence_id"], "evidence_proofs": [ev1, ev2]},
        }
    }
    split_data = {
        "human_event_review_complete": True,
        "event_groups": {"train:1": "g1", "train:2": "g2", "train:3": "g3", "train:10": "g10", "val:0": "gv0"},
        "split": {
            "module_fit": ["train:1", "train:2", "train:3"],
            "temperature_calibration": [],
            "pilot": [],
            "action_train": [],
            "mechanism_validation": ["train:10"],
            "final_eval": ["val:0"],
            "reserve": [],
        },
    }
    adjudications = {
        "T_SRC_DISAGREE": {"adjudicator": "Carol", "adjudicated_label": 1, "rationale": "Confirmed syndicated copy."}
    }
    return ret_a, ret_b, manifest_a, manifest_b, task_objects, split_data, adjudications


def test_converter_roundtrip_loads_into_experiment_runner(tmp_path: Path):
    ret_a, ret_b, ma, mb, objs, split_data, adjs = _fixture_bundle(tmp_path)
    res = convert_returns(ret_a, ret_b, ma, mb, objs, split_data, adjs, strict_complete=True)
    out = tmp_path / "canonical"
    write_conversion(res, out)

    assert res["report"]["exported_counts"] == {"source": 3, "quality": 1, "sufficiency": 0, "source_eval": 1}
    assert res["report"]["resolved_label_breakdown"]["source"] == {
        "positive_1": 2, "negative_0": 0, "unable_to_verify_null": 1
    }
    assert res["report"]["skipped"]["temperature_calibration_excluded"] == ["T_CAL"]

    # Round-trip directly into experiment_runner._load_rows!
    src_rows = _load_rows(out / "source.jsonl", 6)
    qual_rows = _load_rows(out / "quality.jsonl", 3)
    assert len(src_rows) == 3
    assert [r.label.resolve() for r in src_rows] == [1, 1, None]
    assert len(qual_rows) == 1
    assert qual_rows[0].label.resolve() == 1


def test_converter_enforces_security_and_protocol_boundaries(tmp_path: Path):
    ret_a, ret_b, ma, mb, objs, split_data, adjs = _fixture_bundle(tmp_path)

    # 1. Refuses provisional split without explicit flag
    split_unreviewed = {**split_data, "human_event_review_complete": False}
    with pytest.raises(ProtocolError, match="human_event_review_complete"):
        convert_returns(ret_a, ret_b, ma, mb, objs, split_unreviewed, adjs)

    # 2. Refuses adjudicator identical to reviewer A or B
    bad_adj = {"T_SRC_DISAGREE": {"adjudicator": "alice", "adjudicated_label": 1, "rationale": "Self adjudicate"}}
    with pytest.raises(ProtocolError, match="仲裁者不能与甲或乙为同一人"):
        convert_returns(ret_a, ret_b, ma, mb, objs, split_data, bad_adj)

    # 3. Refuses stray adjudication on agreed task
    stray_adj = {**adjs, "T_SRC_AGREE": {"adjudicator": "Carol", "adjudicated_label": 0, "rationale": "Override"}}
    with pytest.raises(ProtocolError, match="双人一致任务不应包含仲裁覆盖"):
        convert_returns(ret_a, ret_b, ma, mb, objs, split_data, stray_adj)

    # 4. Refuses final_eval claim leakage
    objs_leak = json.loads(json.dumps(objs))
    objs_leak["tasks"]["T_QUAL"]["claim_id"] = "val:0"
    with pytest.raises(ProtocolError, match="严禁将官方最终评测集样本"):
        convert_returns(ret_a, ret_b, ma, mb, objs_leak, split_data, adjs)

    # 5. Without adjudication, disagreement is cleanly omitted (not turned into 0) and fails strict_complete
    res_partial = convert_returns(ret_a, ret_b, ma, mb, objs, split_data, adjudications={})
    assert len(res_partial["outputs"]["source"]) == 2
    assert len(res_partial["report"]["skipped"]["unresolved_disagreement"]) == 1
    with pytest.raises(ProtocolError, match="strict-complete"):
        convert_returns(ret_a, ret_b, ma, mb, objs, split_data, adjudications={}, strict_complete=True)
