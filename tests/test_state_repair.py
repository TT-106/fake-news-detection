"""方法修复的真实状态/归档接口回归，使用小型合成夹具，不产生科研成绩。"""
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest

from src.archived_tools import ArchivedEvidenceTools
from src.core_models import SoftLogicRule
from src.faithful_pipeline import ClaimAnalysis, GapResolution, RelationObservation, ToolOutcome
from src.research_contracts import ArchivedPassage, CandidateAction, CostEntry, CostLedger, ProtocolError, ScopedRule, fingerprint
from test_faithful_pipeline import context


def replay_cost(call_id="actual-request-1"):
    return asdict(CostEntry(call_id, "archive-search", .01, 2., True, "request-sha", "actual archived bill"))


def archive_record(costs=None, **outcome):
    return {"costs": [replay_cost()] if costs is None else costs, "outcome": outcome}


def test_analysis_with_numpy_rule_reaches_real_initial_archive_lookup(context, tmp_path):
    _, state, _ = context
    atom = state.atoms[0]
    rule = ScopedRule(SoftLogicRule("rule", "necessary", .2, np.array([1., -1.]), 0.),
                      (atom.atom_id,), atom.entity, atom.event, atom.time, atom.location, "explicit scoped fixture relation")
    analysis = ClaimAnalysis(state.atoms, state.gaps, (rule,))
    # JSON arrays are the pre-existing on-disk representation; numpy objects must match it.
    serial = asdict(analysis)
    serial["rules"][0]["rule"]["c"] = [1., -1.]
    key = fingerprint({"claim": asdict(state.claim), "analysis": serial})
    manifest = tmp_path / "archive.json"
    manifest.write_text(json.dumps({"protocol": "OPEN_WEB_ARCHIVED_REPLAY", "tool_version": "fixture",
                                   "query_template_version": "fixture", "initial_cost_upper_bound": .02,
                                   "initial": {key: archive_record()}}), encoding="utf-8")
    result = ArchivedEvidenceTools(str(manifest)).initial(state.claim, analysis, CostLedger())
    assert result.passages == ()


def test_fingerprint_normalizes_dataclasses_numpy_and_preserves_plain_json(context):
    _, state, _ = context
    plain = {"bool": True, "float": .5, "int": 2, "values": [1, 2], "nested": {"text": "中文"}}
    legacy = hashlib.sha256(json.dumps(plain, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
    converted = {"bool": np.bool_(True), "float": np.float64(.5), "int": np.int64(2),
                 "values": np.array([1, 2]), "nested": {"text": "中文"}}
    assert fingerprint(plain) == legacy == fingerprint(converted)
    assert fingerprint(state.atoms[0]) == fingerprint(asdict(state.atoms[0]))


def test_empty_success_cannot_claim_to_resolve_gap(context):
    engine, state, ledger = context
    action = engine.tools.candidates(state)[0]
    engine.tools.execute = lambda *args: ToolOutcome(resolved_gaps=("date",))
    with pytest.raises(ProtocolError):
        engine.advance(state, action, ledger, .25)


def test_source_verification_cannot_resolve_unrelated_fact_gap(context):
    engine, state, ledger = context
    action = engine.tools.candidates(state)[0]
    engine.tools.execute = lambda *args: ToolOutcome(
        relation=RelationObservation(action.target_pair, True, (state.evidence[0].passage,)),
        resolved_gaps=("date",))
    with pytest.raises(ProtocolError):
        engine.advance(state, action, ledger, .25)


@pytest.mark.parametrize("value", [.37, 0, 1, "true", [], np.bool_(True)])
def test_non_tri_state_relation_rejected_by_archive_parser(context, value):
    _, state, _ = context
    record = archive_record(relation={"target_pair": [0, 1], "same_origin": value,
                                     "proof": [asdict(state.evidence[0].passage)]})
    with pytest.raises(ProtocolError):
        ArchivedEvidenceTools._outcome(record, CostLedger())


def test_state_update_checks_relation_again_even_if_constructor_bypassed(context):
    engine, state, ledger = context
    action = engine.tools.candidates(state)[0]
    relation = RelationObservation(action.target_pair, True, (state.evidence[0].passage,))
    object.__setattr__(relation, "same_origin", .37)
    engine.tools.execute = lambda *args: ToolOutcome(relation=relation)
    with pytest.raises(ProtocolError):
        engine.advance(state, action, ledger, .25)


def test_replayed_cost_preserves_actual_billing_id_and_separate_execution_id():
    ledger = CostLedger()
    ArchivedEvidenceTools._outcome(archive_record(), ledger)
    assert ledger.entries[0].call_id == "actual-request-1"
    assert ledger.entries[0].replayed
    assert ledger.entries[0].replay_execution_id


def test_same_archived_bill_cannot_be_charged_again():
    ledger = CostLedger()
    record = archive_record()
    ArchivedEvidenceTools._outcome(record, ledger)
    with pytest.raises(ProtocolError):
        ArchivedEvidenceTools._outcome(record, ledger)
    assert ledger.total == pytest.approx(.01) and len(ledger.entries) == 1


def test_duplicate_bills_inside_one_record_rejected_before_partial_charge():
    ledger = CostLedger()
    record = archive_record(costs=[replay_cost(), replay_cost()])
    with pytest.raises(ProtocolError):
        ArchivedEvidenceTools._outcome(record, ledger)
    assert ledger.entries == []


def test_invalid_tool_material_still_keeps_actual_request_cost(context):
    _, state, _ = context
    ledger = CostLedger()
    record = archive_record(relation={"target_pair": [0, 1], "same_origin": .37,
                                     "proof": [asdict(state.evidence[0].passage)]})
    with pytest.raises(ProtocolError):
        ArchivedEvidenceTools._outcome(record, ledger)
    assert ledger.total == pytest.approx(.01)
    assert ledger.entries[0].call_id == "actual-request-1"


@pytest.mark.parametrize("value", [float("nan"), np.float64(float("inf")), np.array([np.nan]), object(), {1: "numeric", "1": "string"}])
def test_fingerprint_rejects_nonfinite_and_unsupported_identity(value):
    with pytest.raises(ProtocolError):
        fingerprint(value)


def test_fingerprint_rejects_cyclic_values_and_is_order_stable():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ProtocolError):
        fingerprint(cyclic)
    assert fingerprint({"b": np.array([1., 2.]), "a": ("x",)}) == fingerprint({"a": ["x"], "b": [1., 2.]})
    assert fingerprint({"x", "y"}) == fingerprint(frozenset({"y", "x"}))
    legacy_numeric_keys = {0: (.03, .04), 4: (.01, .02)}
    assert fingerprint(legacy_numeric_keys) == hashlib.sha256(json.dumps(legacy_numeric_keys, sort_keys=True,
        ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("value,expected", [(True, 1.), (False, 0.), (None, .4)])
def test_actual_tri_state_verification_changes_only_target_relation(context, value, expected):
    engine, state, ledger = context
    engine.tools.relation = value
    after = engine.advance(state, engine.tools.candidates(state)[0], ledger, .25)
    assert after.source_matrix[0][1] == expected
    assert after.evidence == state.evidence and after.gaps == state.gaps


def new_passage(tmp_path, evidence_id="e-new"):
    snapshot = tmp_path / f"{evidence_id}.txt"
    text = "The Paris event date was 2020."
    snapshot.write_text(text, encoding="utf-8")
    return ArchivedPassage(evidence_id, f"https://new.example.org/{evidence_id}", str(snapshot),
                           hashlib.sha256(snapshot.read_bytes()).hexdigest(), 0, len(text), text,
                           "2026-11-17T20:00:00+08:00")


def test_only_action_accepted_evidence_with_exact_quote_can_resolve_gap(context, tmp_path):
    engine, state, ledger = context
    passage = new_passage(tmp_path)
    action = CandidateAction("search-date", "fact_search", "Paris event date", .01, .02, "fixture")
    resolution = GapResolution("date", passage.evidence_id, "date was 2020", "The newly retrieved passage states the missing date.")
    engine.tools.execute = lambda *args: ToolOutcome((passage,), resolved_gaps=("date",), gap_resolutions=(resolution,))
    after = engine.advance(state, action, ledger, .25)
    assert after.gaps == () and len(after.evidence) == len(state.evidence)+1
    assert state.gaps == ("date",)


@pytest.mark.parametrize("mismatch", ["existing_id", "unknown_id", "incorrect_quote", "no_binding"])
def test_arbitrary_passage_cannot_unlock_unbound_gap(context, tmp_path, mismatch):
    engine, state, ledger = context
    passage = new_passage(tmp_path)
    action = CandidateAction("search-date", "fact_search", "Paris event date", .01, .02, "fixture")
    evidence_id = {"existing_id": "e0", "unknown_id": "never-obtained"}.get(mismatch, passage.evidence_id)
    quote = "not in the actual passage" if mismatch == "incorrect_quote" else "date was 2020"
    resolutions = () if mismatch == "no_binding" else (GapResolution("date", evidence_id, quote, "Fixture justification"),)
    engine.tools.execute = lambda *args: ToolOutcome((passage,), resolved_gaps=("date",), gap_resolutions=resolutions)
    with pytest.raises(ProtocolError):
        engine.advance(state, action, ledger, .25)


def test_archive_gap_binding_reaches_state_update(context, tmp_path):
    engine, state, ledger = context
    passage = new_passage(tmp_path)
    record = archive_record(passages=[asdict(passage)], resolved_gaps=["date"], gap_resolutions=[{
        "gap": "date", "evidence_id": passage.evidence_id, "quote": "date was 2020", "reason": "Archived exact supporting quote"}])
    engine.tools.execute = lambda current, action, book: ArchivedEvidenceTools._outcome(record, book)
    action = CandidateAction("search-date", "fact_search", "Paris event date", .01, .02, "fixture")
    after = engine.advance(state, action, ledger, .25)
    assert after.gaps == () and any(entry.call_id == "actual-request-1" for entry in ledger.entries)


def test_failure_cannot_use_bound_new_evidence_to_clear_gap(context, tmp_path):
    engine, state, ledger = context
    passage = new_passage(tmp_path)
    engine.tools.execute = lambda *args: ToolOutcome((passage,), resolved_gaps=("date",), failed=True,
        gap_resolutions=(GapResolution("date", passage.evidence_id, "date was 2020", "Fixture"),))
    action = CandidateAction("search-date", "fact_search", "Paris event date", .01, .02, "fixture")
    with pytest.raises(ProtocolError):
        engine.advance(state, action, ledger, .25)
