"""进度账本的证据隔离、不可覆盖与运行归档契约。"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("progress_tool", Path(__file__).resolve().parents[1] / "tools" / "progress.py")
progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progress)


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# frozen source\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "run.json").write_text('{"seed": 42}', encoding="utf-8")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "split.json").write_text('{"eval": ["claim-1", "claim-2"]}', encoding="utf-8")
    (tmp_path / "outputs" / "report.json").write_text('{"n_samples": 2, "metrics": {"macro_f1": 0.5}}', encoding="utf-8")
    (tmp_path / "outputs" / "predictions.jsonl").write_text('{"id":"claim-1","prediction":"true"}\n{"id":"claim-2","prediction":"false"}\n', encoding="utf-8")
    (tmp_path / "progress").mkdir()
    (tmp_path / "progress" / "dashboard.template.html").write_text('<!doctype html><script id="progress-data" type="application/json">__PROGRESS_DATA_JSON__</script>', encoding="utf-8")
    return tmp_path


@pytest.fixture
def completed(workspace):
    return {
        "id": "trial-1", "version": "v1", "title": "Test run", "status": "completed", "kind": "method",
        "started_at": "2026-11-17T01:00:00+08:00", "finished_at": "2026-11-17T01:01:00+08:00", "review_status": "unreviewed",
        "dataset_id": "test", "dataset_revision": "fixture-v1", "split_id": "event-v1", "split_sha256": progress.digest(workspace / "outputs" / "split.json"),
        "protocol_id": "full-eval-v1", "model_id": "frozen-test-model", "budget_id": "budget-1", "n_samples": 2, "seed": 42,
        "command": ["python", "fake-fixture.py"], "config_path": "config/run.json", "metrics": {"macro_f1": 0.5},
        "artifacts": {"config": "config/run.json", "split": "outputs/split.json", "report": "outputs/report.json", "predictions": "outputs/predictions.jsonl"},
        "code_fingerprint": progress.source_fingerprint(workspace)[0],
    }


def test_review_and_withdraw_are_append_only_and_gate_curves(workspace, completed):
    progress.record(completed, workspace)
    original = (workspace / "progress" / "runs" / "trial-1.json").read_bytes()
    assert progress.collect(workspace)["runs"][0]["eligible"] is False
    progress.review("trial-1", "Reviewer", "Predictions and protocol inspected", workspace)
    assert progress.collect(workspace)["runs"][0]["eligible"] is True
    progress.withdraw("trial-1", "event overlap found", workspace)
    derived = progress.collect(workspace)
    assert derived["runs"][0]["withdrawn"] is True
    assert derived["runs"][0]["eligible"] is False
    assert (workspace / "progress" / "runs" / "trial-1.json").read_bytes() == original
    assert len(derived["events"]) == 2


def test_duplicate_id_never_overwrites_history(workspace, completed):
    progress.record(completed, workspace)
    changed = copy.deepcopy(completed)
    changed["version"] = "overwritten"
    with pytest.raises(FileExistsError):
        progress.record(changed, workspace)
    assert progress.collect(workspace)["runs"][0]["version"] == "v1"


def test_changed_or_missing_evidence_excludes_previously_reviewed_run(workspace, completed):
    progress.record(completed, workspace)
    progress.review("trial-1", "Reviewer", "Original evidence inspected", workspace)
    assert progress.collect(workspace)["runs"][0]["eligible"]
    (workspace / "outputs" / "predictions.jsonl").write_text("modified", encoding="utf-8")
    data = progress.collect(workspace)["runs"][0]
    assert not data["eligible"]
    assert data["integrity_issues"]
    (workspace / "outputs" / "predictions.jsonl").unlink()
    assert not progress.collect(workspace)["runs"][0]["eligible"]


def test_comparison_requires_identical_data_model_budget_and_protocol(workspace, completed):
    base = progress.validate_run(completed, workspace)
    other = copy.deepcopy(completed)
    other.update(id="baseline-1", kind="baseline", version="baseline", seed=0)
    assert progress.validate_run(other, workspace)["comparison_key"] == base["comparison_key"]
    for field in ("dataset_revision", "protocol_id", "model_id", "budget_id", "split_id"):
        mismatch = copy.deepcopy(other)
        mismatch[field] += "-different"
        assert progress.validate_run(mismatch, workspace)["comparison_key"] != base["comparison_key"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.01, True])
def test_invalid_metrics_cannot_enter_registry(workspace, completed, value):
    completed["metrics"]["macro_f1"] = value
    with pytest.raises(ValueError):
        progress.record(completed, workspace)
    assert not list((workspace / "progress" / "runs").glob("*.json"))


def test_missing_artifact_traversal_split_mismatch_and_report_mismatch_rejected(workspace, completed):
    for key, value in (("report", "outputs/missing.json"), ("predictions", "../outside.txt")):
        invalid = copy.deepcopy(completed)
        invalid["artifacts"][key] = value
        with pytest.raises(ValueError):
            progress.record(invalid, workspace)
    invalid = copy.deepcopy(completed)
    invalid["split_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="split_sha256"):
        progress.record(invalid, workspace)
    invalid = copy.deepcopy(completed)
    invalid["metrics"]["macro_f1"] = 0.9
    with pytest.raises(ValueError, match="report"):
        progress.record(invalid, workspace)


def test_build_escapes_script_content_and_keeps_stable_content_fingerprint(workspace, completed):
    completed["title"] = '</script><script>alert("injected")</script>'
    progress.record(completed, workspace)
    html = progress.build(workspace).read_text(encoding="utf-8")
    assert completed["title"] not in html
    payload = html.split('type="application/json">')[1].split("</script>")[0]
    assert json.loads(payload)["runs"][0]["title"] == completed["title"]
    before = progress.collect(workspace)["content_fingerprint"]
    progress.build(workspace)
    assert progress.collect(workspace)["content_fingerprint"] == before


def test_wrapper_records_success_without_auto_review(workspace, completed):
    (workspace / "outputs" / "report.json").unlink()
    (workspace / "outputs" / "predictions.jsonl").unlink()
    command = [sys.executable, "-c", write_fixture_outputs() + "; print('fixture completed')"]
    assert progress.execute(completed, command, workspace) == 0
    data = progress.collect(workspace)
    assert data["runs"][0]["status"] == "completed"
    assert data["runs"][0]["review_status"] == "unreviewed"
    assert not data["runs"][0]["eligible"]
    assert data["runs"][0]["code_fingerprint_origin"] == "execution_start"
    archived = workspace / "progress" / "attempts" / "trial-1" / "source" / "src" / "main.py"
    assert archived.read_text(encoding="utf-8") == "# frozen source\n"
    progress.review("trial-1", "Reviewer", "Full trace inspected", workspace)
    assert progress.collect(workspace)["runs"][0]["eligible"]
    archived.write_text("# archive tampered\n", encoding="utf-8")
    assert not progress.collect(workspace)["runs"][0]["eligible"]
    assert (workspace / "progress" / "attempts" / "trial-1" / "stdout.log").read_text().strip() == "fixture completed"


def test_wrapper_records_failed_command_and_keeps_exit_code(workspace, completed):
    (workspace / "outputs" / "report.json").unlink()
    (workspace / "outputs" / "predictions.jsonl").unlink()
    code = progress.execute(completed, [sys.executable, "-c", "import sys; print('failure detail', file=sys.stderr); sys.exit(7)"], workspace)
    assert code == 7
    run = progress.collect(workspace)["runs"][0]
    assert run["status"] == "failed" and run["exit_code"] == 7
    assert run["metrics"] == {} and not run["eligible"]
    assert "failure detail" in (workspace / "progress" / "attempts" / "trial-1" / "stderr.log").read_text()


def test_successful_command_with_bad_report_is_archived_as_failed(workspace, completed):
    (workspace / "outputs" / "report.json").unlink()
    (workspace / "outputs" / "predictions.jsonl").unlink()
    command = [sys.executable, "-c", write_fixture_outputs(include_n=False)]
    code = progress.execute(completed, command, workspace)
    assert code == 2
    run = progress.collect(workspace)["runs"][0]
    assert run["status"] == "failed"
    assert "report.n_samples" in run["failure_reason"]


def test_wrapper_refuses_stale_outputs_before_launch(workspace, completed):
    with pytest.raises(ValueError, match="路径已存在"):
        progress.execute(completed, [sys.executable, "-c", "raise RuntimeError('must not execute')"], workspace)
    assert not list((workspace / "progress" / "attempts").glob("*/start.json"))


def write_fixture_outputs(include_n=True):
    report = {"metrics": {"macro_f1": 0.5}}
    if include_n:
        report["n_samples"] = 2
    predictions = json.dumps([{"sample_id": "claim-1", "prediction": "true"}, {"sample_id": "claim-2", "prediction": "false"}])
    return "from pathlib import Path; import json; Path('outputs/report.json').write_text(" + repr(json.dumps(report)) + "); Path('outputs/predictions.jsonl').write_text('\\n'.join(json.dumps(row) for row in json.loads(" + repr(predictions) + ")))"


def test_record_cannot_self_declare_reviewed(workspace, completed):
    completed["review_status"] = "reviewed"
    with pytest.raises(ValueError, match="具名复核"):
        progress.record(completed, workspace)
    assert not list((workspace / "progress" / "runs").glob("*.json"))


@pytest.mark.parametrize("predictions", ["", "{}\n", '{"sample_id":"one","prediction":"true"}\n', '{"sample_id":"one","prediction":"true"}\n{"sample_id":"one","prediction":"false"}', '{"sample_id":"one","prediction":"true"}\n{"sample_id":"two","prediction":null}'])
def test_predictions_require_full_count_unique_ids_and_nonempty_outputs(workspace, completed, predictions):
    (workspace / "outputs" / "predictions.jsonl").write_text(predictions, encoding="utf-8")
    with pytest.raises(ValueError):
        progress.record(completed, workspace)


@pytest.mark.parametrize("input_name", ["split", "config"])
def test_wrapper_rejects_input_changed_outside_source_fingerprint(workspace, completed, input_name):
    (workspace / "outputs" / "report.json").unlink()
    (workspace / "outputs" / "predictions.jsonl").unlink()
    if input_name == "config":
        (workspace / "outputs" / "execution_config.json").write_text("{}", encoding="utf-8")
        completed["config_path"] = completed["artifacts"]["config"] = "outputs/execution_config.json"
    target = completed["artifacts"][input_name]
    code = write_fixture_outputs() + "; Path(" + repr(target) + ").write_text('changed during execution')"
    assert progress.execute(completed, [sys.executable, "-c", code], workspace) == 2
    run = progress.collect(workspace)["runs"][0]
    assert run["status"] == "failed" and not run["eligible"]
    assert f"原始 {input_name}" in run["failure_reason"]
    assert run["artifacts"][input_name].startswith("progress/attempts/")


def test_experiment_stage_defaults_to_formal_and_separates_groups(workspace, completed):
    formal = progress.validate_run(completed, workspace)
    assert formal["experiment_stage"] == "formal"
    pilot = progress.validate_run(dict(completed, experiment_stage="pilot"), workspace)
    engineering = progress.validate_run(dict(completed, experiment_stage="engineering"), workspace)
    assert len({r["comparison_key"] for r in (formal, pilot, engineering)}) == 3
    with pytest.raises(ValueError, match="experiment_stage"):
        progress.record(dict(completed, experiment_stage="other"), workspace)


@pytest.mark.parametrize("stage,reviewer_kind,eligible", [
    ("formal", "human", True), ("formal", "agent", False),
    ("pilot", "human", True), ("pilot", "agent", True),
    ("engineering", "human", False), ("engineering", "agent", False),
])
def test_review_type_has_stage_specific_eligibility(workspace, completed, stage, reviewer_kind, eligible):
    progress.record(dict(completed, experiment_stage=stage), workspace)
    assert not progress.collect(workspace)["runs"][0]["eligible"]
    event = progress.review("trial-1", "Named reviewer", "Predictions, call logs and recomputed metrics inspected", workspace,
                            reviewer_kind=reviewer_kind)
    assert event["reviewer_kind"] == reviewer_kind
    run = progress.collect(workspace)["runs"][0]
    assert run["eligible"] is eligible and run["reviewer_kind"] == reviewer_kind
    assert run["reviewer"] == "Named reviewer" and run["experiment_stage"] == stage


def test_later_machine_review_does_not_downgrade_prior_human_review(workspace, completed):
    progress.record(completed, workspace)
    progress.review("trial-1", "Human", "Full human evidence review", workspace)
    progress.review("trial-1", "Agent", "Machine run verification", workspace, reviewer_kind="agent")
    run = progress.collect(workspace)["runs"][0]
    assert run["eligible"] and run["reviewer_kind"] == "human" and run["reviewer"] == "Human"


def test_review_kind_invalid_rejected_and_cli_agent_kind_recorded(workspace, completed, monkeypatch):
    progress.record(dict(completed, experiment_stage="pilot"), workspace)
    with pytest.raises(ValueError, match="reviewer_kind"):
        progress.review("trial-1", "Agent", "note", workspace, reviewer_kind="unknown")
    monkeypatch.setattr(progress, "ROOT", workspace)
    assert progress.main(["review", "--id", "trial-1", "--reviewer", "Agent", "--reviewer-kind", "agent",
                          "--note", "Real prediction/log verification performed"]) == 0
    run = progress.collect(workspace)["runs"][0]
    assert run["eligible"] and run["reviewer_kind"] == "agent"


def test_legacy_formal_record_and_review_remain_readable_without_rewriting(workspace, completed):
    stored = progress.record(completed, workspace)
    stored.pop("experiment_stage")
    keys = ("dataset_id", "dataset_revision", "split_id", "split_sha256", "protocol_id", "model_id", "budget_id", "n_samples")
    import hashlib
    stored["comparison_key"] = hashlib.sha256(progress.canonical({k:stored[k] for k in keys}).encode()).hexdigest()
    path = workspace/"progress/runs/trial-1.json"
    progress.write_json(path, stored)
    original = path.read_bytes()
    progress.append_event({"type":"reviewed", "run_id":"trial-1", "reviewer":"Legacy human", "note":"Legacy review"}, workspace)
    run = progress.collect(workspace)["runs"][0]
    assert run["eligible"] and run["experiment_stage"] == "formal" and run["reviewer_kind"] == "human"
    assert run["integrity_issues"] == [] and path.read_bytes() == original


def test_machine_pilot_loses_eligibility_after_withdraw_or_artifact_change(workspace, completed):
    progress.record(dict(completed, experiment_stage="pilot"), workspace)
    progress.review("trial-1", "Agent", "Machine evidence check", workspace, reviewer_kind="agent")
    assert progress.collect(workspace)["runs"][0]["eligible"]
    (workspace/"outputs/predictions.jsonl").write_text("tampered", encoding="utf-8")
    assert not progress.collect(workspace)["runs"][0]["eligible"]
    progress.withdraw("trial-1", "Invalid artifact", workspace)
    run = progress.collect(workspace)["runs"][0]
    assert run["withdrawn"] and not run["eligible"]
