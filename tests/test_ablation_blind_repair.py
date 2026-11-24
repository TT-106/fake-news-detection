"""E2 信息边界、时空遮罩与盲审归档的真实接口回归。"""
from dataclasses import asdict
from hashlib import sha256
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.faithful_pipeline import ClaimAnalysis, FaithfulPipeline, ModelIdentity
from src.input_views import InputView, TextMask
from src.learned_components import ReviewedLabel
from src.mechanism_experiments import evaluate_multimodal_ablations
from src import research_evaluation as evaluation
from src.research_contracts import (ArchivedPassage, AtomicClaim, CanonicalFact, CostLedger,
                                    ProtocolError, PublicClaim, ScoredPassage, TerminalOutput)


def passage(tmp_path):
    directory = tmp_path / "full_ridge" / "model-output"
    directory.mkdir(parents=True)
    path = directory / "source.txt"
    text = "Original report: the Paris event occurred in 2020."
    path.write_text(text, encoding="utf-8")
    return ArchivedPassage("full_ridge-e0", "https://source.example.org/report", str(path),
                           sha256(path.read_bytes()).hexdigest(), 0, len(text), text,
                           "2026-11-17T10:00:00+08:00", "2020-01-01", "Original author", True,
                           "full_ridge/private/provenance.json")


class Binary:
    def __init__(self, purpose, probability):
        self.purpose, self.probability = purpose, probability
        self.version = purpose + "-fixture"
        self.observed = []
    def predict(self, rows):
        self.observed.extend(tuple(row) for row in rows)
        return np.full(len(rows), self.probability)


class VLM:
    identity = ModelIdentity("fixture", "fixture-revision", "fixture-prompt", "fp32")
    def __init__(self):
        self.view = InputView()
        self.view_version = self.view.version
    def with_view(self, view):
        result = copy.copy(self)
        result.view, result.view_version = view, view.version
        return result
    def score(self, claim, atoms, passages, ledger):
        for p in passages:
            self.view.render(claim.claim_id, {"claim": claim.text, "atoms": [asdict(a) for a in atoms], "passage": asdict(p)})
        return tuple(ScoredPassage(p, CanonicalFact("Paris", "event", "occurred", "positive", "2020", "Paris"),
                                   ((1., 1., 1.),), (.9,), (.1,)) for p in passages)
    def terminal(self, state, ledger):
        self.view.render(state.claim.claim_id, {"claim": state.claim.text, "gaps": state.gaps,
                         "evidence": [asdict(e.passage) for e in state.evidence]})
        return TerminalOutput((4., 1., 0., -1.), tuple(e.passage.evidence_id for e in state.evidence), "fixture")


class Temperature:
    temperature, version = 1., "fixture-temperature"
    def probabilities(self, _):
        return (.85, .05, .05, .05)


def test_e2_no_provenance_removes_sufficiency_side_channel(tmp_path):
    p = passage(tmp_path)
    sufficiency = Binary("sufficiency", .9)
    tools = type("Tools", (), {"version": "fixture-tools", "initial_cost_upper_bound": 0.})()
    model = VLM()
    engine = FaithfulPipeline(model, tools, Binary("source", .4), Binary("quality", .8),
                              sufficiency, Temperature(), .6, .01)
    claim = PublicClaim("c", "event-c", "A Paris event", ())
    analysis = ClaimAnalysis((AtomicClaim("a0", "A Paris event", "Paris", "event"),), ("2020 date?",))
    state = engine.assemble_logged(claim, analysis, model.score(claim, analysis.atoms, (p,), CostLedger()), CostLedger())
    review = ReviewedLabel("A", "B", 1, 1, "Complete input mask review; no text removed in this fixture")
    views = {name: InputView(name, {"c": []}, {"c": review}) if name == "no_spatiotemporal" else InputView(name)
             for name in ("full", "no_image", "no_provenance", "no_spatiotemporal", "no_rules")}
    report = evaluate_multimodal_ablations(engine, [state], {"c": "Supported"}, views)
    assert [row[1] for row in sufficiency.observed] == [1., 1., 0., 1., 1.]
    assert report["no_provenance_definition"]["sufficiency_provenance_feature"] == "WITHHELD_ZERO"
    assert state.evidence[0].passage.image_provenance_verified is True


def masked_view(masks):
    return InputView("no_spatiotemporal", {"c": masks},
                     {"c": ReviewedLabel("A", "B", 1, 1, "All time/location text reviewed")})


def test_tuple_and_list_gap_masks_apply_identically():
    view = masked_view([TextMask("gaps.0", 0, 4, "2020")])
    tuple_payload = {"gaps": ("2020 date?",)}
    projected = view.render("c", tuple_payload)
    assert projected == view.render("c", {"gaps": ["2020 date?"]})
    assert projected["gaps"][0] == "[WITHHELD] date?"
    assert tuple_payload["gaps"][0] == "2020 date?"


def test_missing_mask_in_existing_root_is_rejected():
    view = masked_view([TextMask("gaps.9", 0, 4, "2020")])
    with pytest.raises(ProtocolError, match="遮罩.*未应用"):
        view.render("c", {"gaps": ("2020 date?",)})


def test_not_applicable_and_applied_masks_are_traceable():
    view = masked_view([TextMask("gaps.0", 0, 4, "2020")])
    view.render("c", {"claim": "A Paris event"})
    view.render("c", {"gaps": ("2020 date?",)})
    assert view.render_receipts[0]["masks"][0]["status"] == "not_applicable"
    assert view.render_receipts[0]["masks"][0]["reason"] == "payload_root_absent"
    assert view.render_receipts[1]["masks"][0]["status"] == "applied"


def test_evidence_masks_follow_identity_in_both_payload_directions():
    view = masked_view([TextMask("evidence.9.text", 0, 4, "2020", "e1")])
    single = view.render("c", {"passage": {"evidence_id": "e1", "text": "2020 Paris"}})
    full = view.render("c", {"evidence": [{"evidence_id": "e1", "text": "2020 Paris"}]})
    assert single["passage"]["text"] == full["evidence"][0]["text"] == "[WITHHELD] Paris"
    unrelated = view.render("c", {"passage": {"evidence_id": "e2", "text": "2020 Paris"}})
    assert unrelated["passage"]["text"] == "2020 Paris"
    assert view.render_receipts[-1]["masks"][0]["reason"] == "target_evidence_absent"


def test_changed_text_cannot_silently_keep_temporal_signal():
    view = masked_view([TextMask("gaps.0", 0, 4, "2020")])
    with pytest.raises(ProtocolError, match="文本已变化"):
        view.render("c", {"gaps": ("2021 date?",)})
    assert view.render_receipts[-1]["masks"][0]["status"] == "unapplied"


@pytest.mark.parametrize("path", ["claim", "atoms.0.text", "gaps.0", "date", "location"])
def test_global_mask_cannot_be_scoped_to_one_evidence(path):
    with pytest.raises(ProtocolError, match="全局.*证据"):
        masked_view([TextMask(path, 0, 4, "2020", "e1")])


def test_blind_citations_strip_nested_method_identity_but_keep_source(tmp_path):
    p = passage(tmp_path)
    citation = {**asdict(p), "metadata": {"method": "full_ridge", "path": "private/full_ridge"}}
    data = {"c": {"decision": "Supported", "citations": [citation]}}
    packages, key = evaluation.blind_audit_packages(data, data, count=1)
    serialized = json.dumps(packages, ensure_ascii=False)
    assert "full_ridge" not in serialized and "snapshot_path" not in serialized and "metadata" not in serialized
    assert all(x["citations"][0]["text"] == p.text for x in packages)
    assert all(x["citations"][0]["archive_id"].startswith("archive-") for x in packages)
    assert len(key) == 2


def test_blind_neutral_archive_can_replay_after_original_removed(tmp_path):
    p = passage(tmp_path)
    data = {"c": {"decision": "Supported", "citations": [asdict(p)]}}
    archive = tmp_path / "human_audit_materials"
    packages, key = evaluation.blind_audit_packages(data, data, count=1, archive_dir=archive)
    Path(p.snapshot_path).unlink()
    for package in packages:
        citation = package["citations"][0]
        replayed = evaluation.replay_blind_citation(citation, archive)
        assert replayed.text == p.text and replayed.url == p.url
        assert "full_ridge" not in json.dumps(package)
    assert len(list((archive / "snapshots").glob("*.txt"))) == 1


def test_blind_citations_reject_nested_identity_in_whitelisted_scalar(tmp_path):
    p = passage(tmp_path)
    citation = {**asdict(p), "author": {"method": "full_ridge"}}
    data = {"c": {"decision": "Supported", "citations": [citation]}}
    with pytest.raises(ProtocolError, match="嵌套元数据"):
        evaluation.blind_audit_packages(data, data, count=1)


def test_neutral_archive_replay_rejects_modified_body(tmp_path):
    p = passage(tmp_path)
    data = {"c": {"decision": "Supported", "citations": [asdict(p)]}}
    archive = tmp_path / "human_audit_materials"
    packages, _ = evaluation.blind_audit_packages(data, data, count=1, archive_dir=archive)
    citation = packages[0]["citations"][0]
    (archive/citation["archive_file"]).write_text("tampered content", encoding="utf-8")
    with pytest.raises(ProtocolError, match="SHA-256"):
        evaluation.replay_blind_citation(citation, archive)
