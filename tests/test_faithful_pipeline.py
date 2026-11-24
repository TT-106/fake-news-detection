"""小型、明确为合成的协议测试夹具；不会导出为科研实验成绩。"""

from dataclasses import asdict, replace
import copy
import hashlib
import math

import numpy as np
from PIL import Image
import pytest

from src.core_models import LABELS, SoftLogicRule, compute_source_influence_feature
from src.faithful_pipeline import (ClaimAnalysis, FaithfulPipeline, ModelIdentity,
                                    RelationObservation, ToolOutcome, action_features,
                                    choose_one_step, feasible_actions)
from src.learned_components import (ActionGainRegressor, BranchObservation, ExperimentSplit,
                                    ReviewedBinaryModel, ReviewedLabel, SharedTemperature,
                                    SupervisedRow, chow_loss, grouped_folds)
from src.research_contracts import (ArchivedPassage, AtomicClaim, CandidateAction, CanonicalFact,
                                    CostEntry, CostLedger, ImageAsset, Prediction, ProtocolError,
                                    PublicClaim, ScoredPassage, ScopedRule, TerminalOutput)
from src.research_evaluation import (action_regrets, blind_audit_packages, paired_bootstrap,
                                    summarize_predictions)


def bill(ledger, component, amount=.001, success=True):
    ledger.add(CostEntry(f"{component}-{len(ledger.entries)}", component, amount, 1., success,
                         "fixture-sha256", "synthetic unit-test fixture; not a measured experiment"))


class TestBinary:
    __test__ = False
    def __init__(self, purpose, probability):
        self.purpose, self.probability = purpose, probability
        self.version = purpose+"-explicit-test-double"
    def predict(self, values):
        return np.full(len(values), self.probability)


class TestTemperature:
    __test__ = False
    temperature, version = 1., "test-T=1"
    def __init__(self, values=(.8, .1, .05, .05)):
        self.values = values
    def probabilities(self, logits):
        return self.values


class TestVLM:
    __test__ = False
    identity = ModelIdentity("fixture-VLM", "fixture-revision", "fixture-prompt", "fixture-fp32")
    def analyse(self, claim, ledger):
        assert set(asdict(claim)) == {"claim_id", "event_group", "text", "images", "date", "location"}
        bill(ledger, "analyse")
        return ClaimAnalysis((AtomicClaim("a", "A Paris event", "Paris", "event"),), ("date",))
    def score(self, claim, atoms, passages, ledger):
        bill(ledger, "score")
        return tuple(ScoredPassage(p, CanonicalFact("Paris", "event", "occurred", "positive", "2020", "Paris"),
                                   ((1., 1., 1.),), (.9,), (.1,)) for p in passages)
    def terminal(self, state, ledger):
        bill(ledger, "terminal")
        return TerminalOutput((3., 1., 0., -1.), (state.evidence[0].passage.evidence_id,), "fixture only")


class TestTools:
    __test__ = False
    version, initial_cost_upper_bound = "test-tools", .02
    def __init__(self, passages):
        self.passages = passages
        self.relation = True
        self.fail = False
    def initial(self, claim, analysis, ledger):
        bill(ledger, "initial-search", .004)
        return ToolOutcome(self.passages)
    def candidates(self, state):
        return (CandidateAction("source", "source_verify", "verify current sources", .01, .02, "fixture", (0, 1)),)
    def execute(self, state, action, ledger):
        bill(ledger, "source-request", .01, success=not self.fail)
        if self.fail:
            return ToolOutcome(failed=True, error="fixture timeout")
        return ToolOutcome(relation=RelationObservation(action.target_pair, self.relation, (self.passages[0],)))
    def finalize(self, state, ledger):
        bill(ledger, "output", .002)


@pytest.fixture
def context(tmp_path):
    image = tmp_path/"claim.png"
    Image.new("RGB", (8, 8)).save(image)
    claim = PublicClaim("c", "g", "A Paris event", (ImageAsset(str(image), hashlib.sha256(image.read_bytes()).hexdigest()),))
    passages = []
    for i in range(2):
        p = tmp_path/f"snapshot-{i}.txt"
        text = f"A Paris event was recorded in 2020. Source document {i}."
        p.write_text(text, encoding="utf-8")
        passages.append(ArchivedPassage(f"e{i}", f"https://source{i}.example.org/event", str(p),
                                        hashlib.sha256(p.read_bytes()).hexdigest(), 0, len(text), text,
                                        "2026-11-08T10:00:00+08:00"))
    tools = TestTools(tuple(passages))
    engine = FaithfulPipeline(TestVLM(), tools, TestBinary("source", .4), TestBinary("quality", .8),
                              TestBinary("sufficiency", .9), TestTemperature(), .6, .005)
    ledger = CostLedger()
    state = engine.initialise(claim, ledger, .25)
    return engine, state, ledger


def test_formal_claim_whitelists_all_gold_fields(context):
    _, state, _ = context
    sample = {"claim_text": "text", "label": "Refuted", "justification": "SECRET",
              "article": "SECRET_GOLD_URL", "questions": [{"answers": "SECRET"}]}
    claim = PublicClaim.from_sample(sample, "c", "g", state.claim.images)
    assert "SECRET" not in str(asdict(claim))
    assert "questions" not in asdict(claim)


def test_source_action_changes_only_target_P(context):
    engine, state, ledger = context
    action = engine.tools.candidates(state)[0]
    before = copy.deepcopy(state)
    after = engine.advance(state, action, ledger, .25)
    assert after.source_matrix[0][1] == 1.
    assert before.source_matrix == state.source_matrix
    assert after.evidence == state.evidence
    for a, b in zip(after.snippets, state.snippets):
        np.testing.assert_array_equal(a.quality, b.quality)
        np.testing.assert_array_equal(a.r_pos, b.r_pos)
        np.testing.assert_array_equal(a.r_neg, b.r_neg)


def test_unknown_source_relation_stays_unknown(context):
    engine, state, ledger = context
    engine.tools.relation = None
    after = engine.advance(state, engine.tools.candidates(state)[0], ledger, .25)
    assert after.source_matrix == state.source_matrix


def test_failed_action_is_charged_and_adds_no_evidence(context):
    engine, state, ledger = context
    engine.tools.fail = True
    before = ledger.total
    after = engine.advance(state, engine.tools.candidates(state)[0], ledger, .25)
    assert ledger.total == pytest.approx(before+.01)
    failed_request = next(e for e in ledger.entries if e.component == "source-request")
    assert not failed_request.success and failed_request.amount_rmb == .01
    assert after.evidence == state.evidence
    assert after.source_matrix == state.source_matrix


def test_chi_preserves_all_other_inputs(context):
    _, state, _ = context
    original = copy.deepcopy(state)
    value, iota, distance = compute_source_influence_feature(list(state.snippets), np.array(state.source_matrix), (0, 1))
    assert iota == 1 and value == pytest.approx(.4*.6*distance)
    assert state.identity == original.identity
    assert compute_source_influence_feature(list(state.snippets), np.array(state.source_matrix), None) == (0., 0, 0.)


def test_budget_uses_upper_bound_and_output_reserve(context):
    _, state, _ = context
    action = CandidateAction("a", "fact_search", "query", .01, .15, "fixture")
    assert not feasible_actions(state, (action,), .1, .25, .01)
    assert feasible_actions(state, (action,), .09, .25, .01) == (action,)
    assert not feasible_actions(replace(state, step=3), (action,), 0., .25, .01)


@pytest.mark.parametrize("gain", [.0, .05, .1])
def test_stop_is_zero_and_preferred_on_ties(gain):
    action = CandidateAction("a", "fact_search", "query", .1, .11, "fixture")
    assert choose_one_step((action,), {"a": gain}, gamma=1.) == "STOP"
    assert choose_one_step((action,), {"a": .11}, gamma=1.) == "a"


def test_terminal_probability_threshold_is_strict(context):
    engine, state, ledger = context
    engine.temperature.values = (.75, .1, .1, .05)
    p = engine.predict(state, ledger)
    assert p.final_label == "ABSTAIN"
    engine.temperature.values = (.75001, .1, .1, .04999)
    assert not engine.predict(state, ledger).abstained


def test_unknown_citation_cannot_pass_gate(context):
    engine, state, ledger = context
    engine.vlm.terminal = lambda *_: TerminalOutput((4., 0., 0., 0.), ("invented",), "bad fixture")
    result = engine.predict(state, ledger)
    assert result.abstained and not result.citations_replayable


def test_snapshot_tampering_is_detected(context):
    from pathlib import Path
    _, state, _ = context
    p = state.evidence[0].passage
    Path(p.snapshot_path).write_text("CHANGED CONTENT", encoding="utf-8")
    with pytest.raises(ProtocolError, match="SHA-256"):
        p.verify()


def test_chow_gain_retains_negative_zero_and_positive():
    rows = [BranchObservation("c", "g", "h", "a", (), a, b, .01, .01)
            for a,b in [(0., 1.), (.25, .25), (1., 0.)]]
    assert [r.gain for r in rows] == [-1., 0., 1.]
    with pytest.raises(ProtocolError):
        _ = replace(rows[0], loss_before=.37).gain


def test_common_history_gains_are_exact_chow_differences(context):
    engine, state, ledger = context
    rows = engine.common_history_branches(state, ledger, "Supported")
    assert {r.history_id for r in rows} == {state.identity}
    assert all(r.gain == r.loss_before-r.loss_after for r in rows)
    assert next(r for r in rows if r.action_id == "STOP").gain == 0.


def test_no_generic_rule_suppresses_evidence_conflict(context):
    _, state, _ = context
    rule = ScopedRule(SoftLogicRule("bad", "incompatibility", 1., np.array([1., 1.]), -1.),
                      ("a",), "Paris", "event", "", "", "not a valid relational rule")
    with pytest.raises(ProtocolError, match="证据冲突"):
        rule.validate(state.atoms)


def test_fact_identity_has_all_temporal_qualifiers():
    a = CanonicalFact("A", "E", "R", "+", "2020", "Paris")
    assert replace(a, time="2021").key != a.key
    assert replace(a, location="London").key != a.key
    assert replace(a, event="other event").key != a.key
    assert replace(a, polarity="-").key != a.key


def test_without_chi_changes_only_chi_block(context):
    engine, state, ledger = context
    a = engine.tools.candidates(state)[0]
    full = action_features(state, a, ledger.total, .25)
    without = action_features(state, a, ledger.total, .25, mode="no_chi")
    assert full[:-1] == without
    # 无目标行动 χ=0，与目标存在但影响恰好为零由 iota 区分。
    search = CandidateAction("search", "fact_search", "query", .01, .02, "fixture")
    assert action_features(state, search, ledger.total, .25)[-1] == 0.


def test_fit_split_rejects_event_overlap():
    split = ExperimentSplit(frozenset({"fit"}), frozenset({"cal"}), frozenset(), frozenset(), frozenset(), frozenset())
    with pytest.raises(ProtocolError, match="事件组"):
        split.validate({"fit": "same-event", "cal": "same-event"})


def test_review_requires_independent_reviewers_and_adjudication():
    with pytest.raises(ProtocolError):
        ReviewedLabel("same", "same", 1, 1, "reason").resolve()
    with pytest.raises(ProtocolError):
        ReviewedLabel("a", "b", 1, 0, "reason").resolve()
    assert ReviewedLabel("a", "b", None, None, "unverifiable").resolve() is None
    assert ReviewedLabel("a", "b", 1, 0, "reason", "c", 0).resolve() == 0


def test_binary_models_do_not_train_on_final_eval():
    split = ExperimentSplit(frozenset({"fit"}), frozenset(), frozenset(), frozenset(), frozenset(), frozenset({"eval"}))
    row = SupervisedRow("eval", "eval-group", (1.,), ReviewedLabel("a", "b", 1, 1, "reason"))
    with pytest.raises(ProtocolError, match="开发拟合集"):
        ReviewedBinaryModel("source").fit([row], split, {"fit": "fit-group", "eval": "eval-group"})


def test_grouped_folds_never_split_histories_from_one_event():
    groups = np.repeat(["a", "b", "c", "d"], 4)
    y = np.tile([0, 0, 1, 1], 4)
    for train, val in grouped_folds(groups, y, True):
        assert not set(groups[train]) & set(groups[val])


def test_no_coverage_has_undefined_selective_risk():
    p = Prediction((.25,)*4, "Refuted", "ABSTAIN", .1, (), False, "s")
    result = summarize_predictions({"c": p}, {"c": "Supported"}, {"c": .01})
    assert result["coverage"] == 0 and result["selective_risk"] is None
    assert result["chow_risk"] == .25 and result["macro_f1"] == 0


def test_abstentions_count_in_primary_macro_f1():
    a = Prediction((.8,.1,.05,.05), "Supported", "Supported", .9, ("e",), True, "s")
    b = replace(a, final_label="ABSTAIN")
    result = summarize_predictions({"a": a, "b": b}, {"a": "Supported", "b": "Supported"}, {"a": 0., "b": 0.})
    assert result["macro_f1"] == pytest.approx((2/3)/4)
    assert result["selective_macro_f1"] == .25


def test_bootstrap_aligns_claim_ids_and_rejects_missing_pairs():
    assert paired_bootstrap({"a": 1., "b": 0.}, {"b": 0., "a": 1.}, n_boot=100)["95ci"] == [0., 0.]
    with pytest.raises(ProtocolError):
        paired_bootstrap({"a": 1.}, {"b": 1.})


def test_blind_audits_never_invent_human_scores():
    data = {"c": {"decision": "Supported", "citations": ["e"], "method_name": "SECRET", "score": 1.}}
    packages, key = blind_audit_packages(data, data, count=1)
    assert len(packages) == len(key) == 2
    assert all(p["human_review_a"] is None and p["human_review_b"] is None for p in packages)
    assert "SECRET" not in str(packages) and "score" not in str(packages)


def test_fitted_module_versions_detect_temperature_mutation(context):
    engine, state, ledger = context
    engine.temperature.temperature = 2.
    with pytest.raises(ProtocolError, match="冻结"):
        engine.predict(state, ledger)


def test_full_episode_counts_initial_failure_processing_and_output_costs(context):
    engine, state, _ = context
    engine.tools.fail = True
    class Gain:
        feature_mode = "full"
        def predict_gain(self, _):
            return .1
    result = engine.run_episode(state.claim, Gain(), budget=.25)
    assert result["total_cost_rmb"] == pytest.approx(.02)
    assert any(not e["success"] for e in result["ledger"])
    assert {e["component"] for e in result["ledger"]} >= {"analyse", "initial-search", "score", "terminal", "source-request", "output"}
    assert result["trajectory"][0]["action"] == "source"
    assert result["trajectory"][-1]["action"] == "STOP"


def test_binary_model_actually_fits_reviewed_data_without_eval_labels():
    fit = frozenset({"f1", "f2", "f3", "f4"})
    groups = {c: c+"-event" for c in fit}
    split = ExperimentSplit(fit, frozenset(), frozenset(), frozenset(), frozenset(), frozenset())
    rows = [SupervisedRow(c, groups[c], (float(y), .2), ReviewedLabel("reviewer-A", "reviewer-B", y, y, "fixture review"))
            for c in sorted(fit) for y in (0, 1)]
    model = ReviewedBinaryModel("source").fit(rows, split, groups)
    assert model.training_claims == fit
    assert model.predict([(0., .2), (1., .2)])[1] > model.predict([(0., .2), (1., .2)])[0]
    old = model.version
    model.estimator["model"].coef_[0,0] += 1.
    assert model.version != old


def test_temperature_fits_only_independent_groups():
    from src.learned_components import CalibrationRow
    split = ExperimentSplit(frozenset({"fit"}), frozenset({"cal1", "cal2"}), frozenset(), frozenset(), frozenset(), frozenset({"eval"}))
    groups = {c: c+"-event" for c in ("fit", "cal1", "cal2", "eval")}
    rows = [CalibrationRow("cal1", groups["cal1"], (2., 0., -1., -1.), LABELS[0]),
            CalibrationRow("cal2", groups["cal2"], (0., 2., -1., -1.), LABELS[1])]
    temperature = SharedTemperature().fit(rows, split, groups)
    assert temperature.temperature > 0 and sum(temperature.probabilities(rows[0].logits)) == pytest.approx(1.)
    with pytest.raises(ProtocolError):
        SharedTemperature().fit([replace(rows[0], claim_id="eval", event_group=groups["eval"])], split, groups)


def test_gain_regression_keeps_real_negative_and_zero_examples():
    ids = frozenset({"t1", "t2", "t3"})
    split = ExperimentSplit(frozenset(), frozenset(), frozenset(), ids, frozenset(), frozenset())
    groups = {c: c+"-event" for c in ids}
    rows = [BranchObservation(c, groups[c], c+"h", "action", (float(i),), before, after, .01, .01)
            for c in sorted(ids) for i,(before,after) in enumerate([(0.,1.),(.25,.25),(1.,0.)])]
    model = ActionGainRegressor().fit(rows, split, groups, "fixture-frozen-modules")
    assert math.isfinite(model.predict_gain((1.,)))
    assert len(rows) == 9 and {r.gain for r in rows} == {-1.,0.,1.}


def test_source_evaluation_excludes_unknown_labels(context):
    from src.mechanism_experiments import SourceEvalPair, evaluate_source_methods
    engine, state, _ = context
    engine.source.training_groups = frozenset({"independent-training-event"})
    pair = SourceEvalPair("c", "e0", "e1", ReviewedLabel("A", "B", None, None, "cannot verify"),
                          (state.evidence[0].passage,))
    report = evaluate_source_methods([state], [pair], engine.source, {"c"})
    assert report["verified_eval_pairs"] == 0 and report["source_brier_score"] is None
    assert report["records"][0]["oracle_state_drift"] is None


def test_real_source_upper_bound_uses_reviewed_relations(context):
    from src.mechanism_experiments import SourceEvalPair, evaluate_source_methods
    engine, state, _ = context
    engine.source.training_groups = frozenset({"training-event"})
    pair = SourceEvalPair("c", "e0", "e1", ReviewedLabel("A", "B", 1, 1, "archived reference chain"),
                          (state.evidence[0].passage,))
    report = evaluate_source_methods([state], [pair], engine.source, {"c"})
    assert report["source_brier_score"] == pytest.approx((.4-1)**2)
    assert report["records"][0]["bound_satisfied"]


def test_no_image_and_provenance_are_actual_input_views():
    from src.input_views import InputView
    assert not InputView("no_image").include_images
    payload = {"claim": "text", "passage": {"text": "real text", "url": "https://example.org", "author": "A", "published_at": "2020"}}
    projected = InputView("no_provenance").render("c", payload)
    assert projected["passage"] == {"text": "real text"}
    assert payload["passage"]["author"] == "A"


def test_spatiotemporal_masks_verify_text_and_follow_evidence_ids():
    from src.input_views import InputView, TextMask
    review = ReviewedLabel("A", "B", 1, 1, "complete mask reviewed")
    view = InputView("no_spatiotemporal", {"c": [TextMask("passage.text", 0, 4, "2020", "e1")]}, {"c": review})
    single = view.render("c", {"passage": {"evidence_id": "e1", "text": "2020 Paris"}})
    terminal = view.render("c", {"evidence": [{"evidence_id": "e0", "text": "other"}, {"evidence_id": "e1", "text": "2020 Paris"}]})
    assert single["passage"]["text"] == terminal["evidence"][1]["text"] == "[WITHHELD] Paris"
    with pytest.raises(ProtocolError):
        view.render("c", {"passage": {"evidence_id": "e1", "text": "2021 Paris"}})


def test_missing_model_cannot_silently_produce_formal_scores(tmp_path, monkeypatch):
    import json
    from src import real_averimatec_pipeline as entry
    data = tmp_path/"data/averimatec"
    data.mkdir(parents=True)
    (data/"train.json").write_text(json.dumps([{"claim_text": "Train claim", "label": "Refuted"}]), encoding="utf-8")
    (data/"val.json").write_text(json.dumps([{"claim_text": "Unrelated held out test", "label": "Supported"}]), encoding="utf-8")
    monkeypatch.setattr(entry, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(entry, "get_nli_model", lambda: pytest.fail("preflight must not load an alternative model"))
    report = entry.check_readiness(tmp_path/"missing.json")
    assert report["formal_scores"] is None and not report["formal_experiments_executed"]
    assert report["status"] == "BLOCKED_MISSING_PREREQUISITES"


def test_permutation_uses_only_frozen_training_strata():
    from src.learned_components import MatchedChiPermutation
    rows = [BranchObservation(f"t{i}", f"g{i}", f"h{i}", "source", (0.,)*14+(1.,chi),
                               .25,.25,.01,.01,u)
            for i,(u,chi) in enumerate([(.24,.01),(.24,.02),(.01,.03),(.01,.04)])]
    control = MatchedChiPermutation(seed=42)
    permuted = control.fit_transform(rows)
    assert sorted(r.features[-1] for r in permuted[:2]) == [.01,.02]
    assert sorted(r.features[-1] for r in permuted[2:]) == [.03,.04]
    assert control.donor("UNSEEN_EVAL_HISTORY", "source", .24) in {.01,.02}
    assert control.donor("UNSEEN_EVAL_HISTORY", "source", .24) == control.donor("UNSEEN_EVAL_HISTORY", "source", .24)
    assert control.version


def test_temporal_image_mask_changes_only_model_visible_copy():
    from src.input_views import ImageMask, InputView, TextMask
    original = Image.new("RGB", (8,8), "white")
    review = ReviewedLabel("A", "B", 1, 1, "text and image signals reviewed")
    view = InputView("no_spatiotemporal", {"c": []}, {"c": review},
                     {"sha": ImageMask("sha",8,8,((0,0,4,4),))})
    masked = view.project_image("sha", original)
    assert masked.getpixel((1,1)) == (0,0,0) and masked.getpixel((7,7)) == (255,255,255)
    assert original.getpixel((1,1)) == (255,255,255)
    with pytest.raises(ProtocolError):
        view.project_image("unreviewed-image", original)
    masked.close()
    original.close()


def test_first_terminal_call_is_part_of_initial_cost_cap(context):
    engine, state, _ = context
    # analyse + retrieval + score = .006，首次终端评分另需 .001。
    engine.tools.initial_cost_upper_bound = .0065
    class Stop:
        feature_mode = "no_chi"
        def predict_gain(self, _):
            return 0.
    with pytest.raises(ProtocolError, match="首次终端"):
        engine.run_episode(state.claim, Stop(), .25)
