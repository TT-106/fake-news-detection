import json

import run_all_preliminary_experiments as preliminary
from src import real_averimatec_pipeline
from src.data_integrity import AI_EVENT_PROTOCOL


def test_clean_checkout_reports_missing_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(real_averimatec_pipeline, "ROOT_DIR", tmp_path)

    status = preliminary.main(["--output-dir", str(tmp_path / "report")])

    report = json.loads((tmp_path / "report" / "preflight.json").read_text(encoding="utf-8"))
    assert status == 2
    assert any(item["status"] == "MISSING_AUDIT_INPUT" for item in report["blockers"])


def test_sealed_audit_preserves_historical_provisional_split(monkeypatch, tmp_path):
    audit = {
        "protocol": AI_EVENT_PROTOCOL,
        "isolation_status": "PASS_FOR_OBSERVED_KEYS",
        "val_file_opened": False,
    }
    monkeypatch.setattr(real_averimatec_pipeline, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(real_averimatec_pipeline, "audit_data", lambda root: audit)
    historical_split = tmp_path / "provisional_split.json"
    historical_split.write_text("historical split", encoding="utf-8")

    status = preliminary.main(["--output-dir", str(tmp_path)])

    assert status == 2
    assert historical_split.read_text(encoding="utf-8") == "historical split"
    assert json.loads((tmp_path / "sealed_dataset_audit.json").read_text(encoding="utf-8")) == audit
