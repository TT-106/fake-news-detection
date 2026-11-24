"""针对已复现的标签泄漏、虚构证据和数学前提缺失的回归检查。"""

import copy

import numpy as np
import pytest

from src.core_models import EvidenceSnippet, SoftLogicRule, solve_convex_qp_inference
from src import real_averimatec_pipeline as pipeline


def sample():
    return {
        "claim_text": "This photograph shows the London flood in 2023.",
        "date": "2023-06-01", "claim_images": [], "label": "Refuted",
        "justification": "SECRET_GOLD_JUSTIFICATION",
        "questions": [{
            "question": "When was the photograph first published?",
            "answer_method": "Web search", "question_type": ["Text"],
            "answers": [{"answer_type": "Extractive", "answer_text": "Published in 1995.",
                         "source_url": "https://archive.example.org/photo/1995"}],
        }],
    }


@pytest.fixture(autouse=True)
def deterministic_stance(monkeypatch):
    class Stance:
        def stance(self, premise, hypothesis):
            return 0.1, 0.8, 0.1
    monkeypatch.setattr(pipeline, "get_nli_model", lambda: Stance())


def test_ground_truth_cannot_change_evidence():
    from src.exp3_pilot_20_active_verification import extract_sample_evidence_and_source_pairs
    original = sample()
    changed = copy.deepcopy(original)
    changed["label"] = "Supported"
    changed["justification"] = "DIFFERENT_GOLD_JUSTIFICATION"
    a = extract_sample_evidence_and_source_pairs(original, 0, np.random.default_rng(42))
    b = extract_sample_evidence_and_source_pairs(changed, 0, np.random.default_rng(42))
    assert len(a["snippets"]) == len(b["snippets"])
    for x, y in zip(a["snippets"], b["snippets"]):
        np.testing.assert_array_equal(x.r_pos, y.r_pos)
        np.testing.assert_array_equal(x.r_neg, y.r_neg)


def test_native_evidence_is_neither_duplicated_nor_fabricated():
    env = pipeline.build_claim_evidence_environment(sample(), 0)
    assert len(env["raw_records"]) == 1
    assert env["raw_records"][0]["url"] == "https://archive.example.org/photo/1995"


def test_empty_evidence_stays_empty():
    s = sample()
    s["questions"] = []
    env = pipeline.build_claim_evidence_environment(s, 0)
    assert env["raw_records"] == []
    assert env["snippets"] == []


def test_a_url_is_not_proof_of_citation_replay():
    env = pipeline.build_claim_evidence_environment(sample(), 0)
    assert not env["snippets"][0].url_replayable


def test_fact_identity_is_not_a_time_or_location_slot():
    s = sample()
    q = copy.deepcopy(s["questions"][0])
    q["question"] = "When did the unrelated bridge open?"
    q["answers"][0]["answer_text"] = "The bridge opened in 2010."
    q["answers"][0]["source_url"] = "https://archive.example.org/bridge/2010"
    s["questions"].append(q)
    env = pipeline.build_claim_evidence_environment(s, 0)
    assert env["snippets"][0].fact_id != env["snippets"][1].fact_id


def snippet():
    return EvidenceSnippet(0, "example.org", "https://example.org/1", 1,
                           np.array([0.8]), np.array([0.9]), np.array([0.1]))


@pytest.mark.parametrize("mu", [0.0, -0.1, np.nan])
def test_strong_convexity_is_required(mu):
    with pytest.raises(ValueError):
        solve_convex_qp_inference([snippet()], np.eye(1), mu=mu)


def test_asymmetric_source_matrix_is_rejected():
    s1, s2 = snippet(), snippet()
    s2.snippet_id = 1
    with pytest.raises(ValueError):
        solve_convex_qp_inference([s1, s2], np.array([[1.0, 0.2], [0.8, 1.0]]))


def test_negative_rule_weight_is_not_a_convex_objective():
    rule = SoftLogicRule("invalid", "implication", -1.0, np.array([1., -1.]), 0.)
    with pytest.raises(ValueError):
        solve_convex_qp_inference([snippet()], np.eye(1), rules=[rule])
