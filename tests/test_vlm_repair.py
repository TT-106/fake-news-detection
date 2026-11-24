"""S1/E2 回归：只用明确合成夹具，不等价于真实模型/规则标注验收。"""
from dataclasses import asdict
from contextlib import nullcontext
import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.faithful_pipeline import FaithfulPipeline
from src.research_contracts import (ArchivedPassage, CanonicalFact, CostLedger,
                                    ProtocolError, PublicClaim, ScoredPassage, fingerprint)
from src.vlm_backend import LocalQwenVLM
from src import vlm_backend


@pytest.fixture
def fake_runtime(monkeypatch):
    model = SimpleNamespace(eval=lambda: model, parameters=lambda: ())
    processor = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda s, **kw: [ord(s)]))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True), bfloat16="fixture-bf16"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: processor),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=lambda *a, **kw: model)))


@pytest.fixture
def materials(tmp_path, fake_runtime):
    directory = tmp_path / "fixture-model"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({
        "model_type": "qwen2_5_vl", "hidden_size": 3584, "num_hidden_layers": 28}), encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"explicit non-model unit fixture")
    (directory / "chat_template.jinja").write_text("template-A", encoding="utf-8")
    nested = directory / "processor-assets"
    nested.mkdir()
    (nested / "chat.jinja").write_text("nested-template-A", encoding="utf-8")
    (nested / "vocabulary.model").write_bytes(b"fixture tokenizer resource A")
    claim = PublicClaim("fixture:0", "fixture-event", "A person drove a car in Paris in 2020.", (), "2020", "Paris")
    context = dict(entity="person", event="car journey", time="2020", location="Paris")
    atoms = [dict(atom_id="a0", text="The person drove a car.", **context),
             dict(atom_id="a1", text="The person was in a vehicle.", **context)]
    proof_path = tmp_path / "relation-proof.txt"
    proof_text = "SYNTHETIC TEST FIXTURE: driving this car requires being in a vehicle."
    proof_path.write_text(proof_text, encoding="utf-8")
    proof = ArchivedPassage("fixture-proof", "https://fixture.example/relation", str(proof_path),
                            hashlib.sha256(proof_path.read_bytes()).hexdigest(), 0, len(proof_text),
                            proof_text, "2026-11-17T10:00:00+08:00")
    digest = fingerprint({"claim_id": claim.claim_id, "text": claim.text, "date": claim.date,
                          "location": claim.location, "image_sha256": []})
    record = {"rule_id": "fixture-r0", "claim_id": claim.claim_id, "claim_input_sha256": digest,
              "rule_type": "implication", "antecedent": {k:v for k,v in atoms[0].items() if k != "atom_id"},
              "consequent": {k:v for k,v in atoms[1].items() if k != "atom_id"}, "weight": 1.0,
              "justification": "Explicit synthetic reviewed relation; not a research annotation.",
              "review": {"reviewer_a": "fixture-human-a", "reviewer_b": "fixture-human-b", "label_a": 1,
                         "label_b": 1, "rationale": "Synthetic fixture only; no real humans are represented."},
              "proofs": [asdict(proof)]}
    manifest = {"protocol": "PREVERIFIED_SCOPED_RULES_V1", "supervision_scope": "logical_relation_only",
                "records": [record]}
    path = tmp_path / "fixture-rules.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return SimpleNamespace(directory=directory, tmp=tmp_path, claim=claim, atoms=atoms,
                           proof=proof, manifest=manifest, manifest_path=path)


def build(materials, manifest=True):
    kwargs = {"rule_manifest_path": str(materials.manifest_path)} if manifest else {}
    return LocalQwenVLM(str(materials.directory), "bf16", str(materials.tmp / "calls"), **kwargs)


def analyse(vlm, materials, rules=None):
    captured = []
    def call(operation, claim, payload, ledger, **kwargs):
        captured.append(payload)
        return {"atoms": copy.deepcopy(materials.atoms), "gaps": [], "rules": rules if rules is not None else [
            {"rule_id": "fixture-r0", "antecedent_atom_id": "a0", "consequent_atom_id": "a1"}]}
    vlm._call = call
    return vlm.analyse(materials.claim, CostLedger()), captured


def rewrite(materials):
    materials.manifest_path.write_text(json.dumps(materials.manifest), encoding="utf-8")


def test_formal_analyse_receives_catalogue_and_builds_scoped_rule(materials):
    parsed, captured = analyse(build(materials), materials)
    assert [r["rule_id"] for r in captured[0]["rule_catalogue"]] == ["fixture-r0"]
    assert len(parsed.rules) == 1
    rule = parsed.rules[0]
    assert rule.atom_ids == ("a0", "a1")
    np.testing.assert_array_equal(rule.rule.c, [1, -1, 0, 0])
    rule.validate(parsed.atoms)


@pytest.mark.parametrize("kind,expected,d", [
    ("implication", [1,-1,0,0], 0),
    ("necessary", [0,0,-1,1], 0),
    ("incompatibility", [1,1,0,0], -1),
])
def test_three_relation_types_have_fixed_auditable_semantics(materials, kind, expected, d):
    materials.manifest["records"][0]["rule_type"] = kind
    rewrite(materials)
    parsed, _ = analyse(build(materials), materials)
    np.testing.assert_array_equal(parsed.rules[0].rule.c, expected)
    assert parsed.rules[0].rule.d == d


def test_formal_state_with_rule_differs_from_no_rules_on_same_evidence(materials):
    parsed, _ = analyse(build(materials), materials)
    engine = FaithfulPipeline.__new__(FaithfulPipeline)
    engine.quality = SimpleNamespace(predict=lambda x: np.full(len(x), .8))
    engine.source = SimpleNamespace(predict=lambda x: np.full(len(x), .4))
    passage = materials.proof
    evidence = (ScoredPassage(passage, CanonicalFact("person", "car journey", "driving", "positive", "2020", "Paris"),
                              ((1.,1.,1.),(1.,1.,1.)), (.9,.1), (.1,.1)),)
    full = engine._assemble(materials.claim, parsed, evidence)
    no_rules = engine._assemble(materials.claim, type(parsed)(parsed.atoms, parsed.gaps), evidence)
    assert len(full.rules) == 1 and not no_rules.rules
    assert not np.allclose(full.z, no_rules.z)
    assert full.identity != no_rules.identity


def test_no_catalogue_does_not_silently_accept_raw_model_rules(materials):
    with pytest.raises(ProtocolError, match="规则|rule"):
        analyse(build(materials, manifest=False), materials)


def test_no_catalogue_and_no_rule_is_explicitly_allowed(materials):
    parsed, captured = analyse(build(materials, manifest=False), materials, [])
    assert not parsed.rules
    assert captured[0]["rule_catalogue"] == []


@pytest.mark.parametrize("proposal", [
    [{"rule_id":"not-approved", "antecedent_atom_id":"a0", "consequent_atom_id":"a1"}],
    [{"rule_id":"fixture-r0", "antecedent_atom_id":"a1", "consequent_atom_id":"a0"}],
    [{"rule_id":"fixture-r0", "antecedent_atom_id":"a0", "consequent_atom_id":"a1", "c":[1,0,0,0]}],
    ["fixture-r0"],
])
def test_unapproved_misbound_or_arbitrary_coefficient_rule_is_rejected(materials, proposal):
    with pytest.raises(ProtocolError):
        analyse(build(materials), materials, proposal)


@pytest.mark.parametrize("change", ["review", "proof", "scope", "weight", "duplicate"])
def test_unreviewed_tampered_or_unscoped_catalogue_is_rejected(materials, change):
    record = materials.manifest["records"][0]
    if change == "review": record["review"]["reviewer_b"] = record["review"]["reviewer_a"]
    if change == "proof": record["proofs"][0]["snapshot_sha256"] = "0"*64
    if change == "scope": record["consequent"]["time"] = "2021"
    if change == "weight": record["weight"] = -1
    if change == "duplicate": materials.manifest["records"].append(copy.deepcopy(record))
    rewrite(materials)
    with pytest.raises(ProtocolError): build(materials)


def test_catalogue_is_bound_to_actual_public_claim(materials):
    vlm = build(materials)
    materials.claim = PublicClaim(materials.claim.claim_id, "fixture-event", "A different public claim.", (), "2020", "Paris")
    with pytest.raises(ProtocolError, match="版本|声明|绑定"):
        analyse(vlm, materials)


def test_proof_is_revalidated_and_rejection_is_traceable(materials):
    vlm = build(materials)
    materials.proof_path = materials.tmp / "relation-proof.txt"
    materials.proof_path.write_text("tampered after model freeze", encoding="utf-8")
    with pytest.raises(ProtocolError):
        analyse(vlm, materials)
    logs = list((materials.tmp / "calls").glob("analysis-check-*.json"))
    assert len(logs) == 1
    assert json.loads(logs[0].read_text(encoding="utf-8"))["status"] == "REJECTED"


def test_model_invented_rule_is_recorded_as_rejection(materials):
    with pytest.raises(ProtocolError):
        analyse(build(materials, manifest=False), materials)
    logs = list((materials.tmp / "calls").glob("analysis-check-*.json"))
    assert json.loads(logs[0].read_text(encoding="utf-8"))["status"] == "REJECTED"


@pytest.mark.parametrize("resource", ["chat_template.jinja", "processor-assets/chat.jinja", "processor-assets/vocabulary.model"])
def test_constructor_identity_changes_for_actual_processor_resource(materials, resource):
    before = build(materials, manifest=False).identity
    p = materials.directory / resource
    p.write_bytes(p.read_bytes()+b" changed")
    after = build(materials, manifest=False).identity
    assert before.revision != after.revision


def test_reviewed_catalogue_is_part_of_frozen_identity(materials):
    before = build(materials).identity
    materials.manifest["records"][0]["weight"] = .5
    rewrite(materials)
    after = build(materials).identity
    assert before != after


@pytest.fixture
def bnb_constructor(materials, monkeypatch):
    """只替代原生权重加载，仍经过正式构造及量化配置接口。"""
    observed = []
    module = sys.modules["transformers"]
    factory = module.AutoModelForImageTextToText.from_pretrained
    class QuantizationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    def load(*args, **kwargs):
        observed.append(kwargs)
        return factory(*args, **kwargs)
    monkeypatch.setattr(module, "BitsAndBytesConfig", QuantizationConfig, raising=False)
    monkeypatch.setattr(module.AutoModelForImageTextToText, "from_pretrained", load)
    def construct():
        model = LocalQwenVLM(str(materials.directory), "bnb4", str(materials.tmp / "bnb-calls"))
        return model, observed[-1]
    return construct


def test_bnb4_constructor_explicitly_uses_nf4_double_quant_bf16(bnb_constructor):
    model, received = bnb_constructor()
    assert received["quantization_config"].kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "fixture-bf16",
    }
    assert received["local_files_only"] is True
    assert received["device_map"] == "auto"
    assert received["dtype"] == "fixture-bf16"
    assert model.identity.quantization == "bnb4"


@pytest.mark.parametrize("key,replacement", [
    ("bnb_4bit_quant_type", "fp4"),
    ("bnb_4bit_use_double_quant", False),
])
def test_bnb4_actual_loading_settings_are_frozen_in_identity(bnb_constructor, monkeypatch, key, replacement):
    before, _ = bnb_constructor()
    changed = {**vlm_backend.BNB4_LOADING_SETTINGS, key: replacement}
    monkeypatch.setattr(vlm_backend, "BNB4_LOADING_SETTINGS", changed)
    after, received = bnb_constructor()
    assert received["quantization_config"].kwargs[key] == replacement
    assert before.identity.quantization == after.identity.quantization == "bnb4"
    assert before.identity.prompt_sha256 != after.identity.prompt_sha256


def test_call_uses_multimodal_content_blocks_for_system_and_user(materials):
    model = build(materials, manifest=False)
    class Inputs(dict):
        def to(self, device):
            return self
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)
        def __getitem__(self, key):
            return Tensor(self.values[key])
        def float(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.values
    observed = []
    def process(messages, **kwargs):
        # 与真实 Transformers 4.57.6 处理器一致，逐 content block 读取 type。
        for message in messages:
            observed.append([content["type"] for content in message["content"]])
        return Inputs(input_ids=SimpleNamespace(shape=(1, 5)))
    class Forward:
        device = "fixture-cpu"
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=Tensor([[[.2, .7, .1, 0.]]]))
    model.processor.apply_chat_template = process
    model.model = Forward()
    model.torch = SimpleNamespace(inference_mode=nullcontext)
    model.class_token_ids = [0, 1, 2, 3]
    values = model._call("terminal", materials.claim, {"claim": materials.claim.text}, CostLedger(), logits=True)
    assert observed == [["text"], ["text"]]
    np.testing.assert_allclose(values, [.2, .7, .1, 0.])


def test_processor_constructor_explicitly_locks_checkpoint_slow_path(materials, monkeypatch):
    module = sys.modules["transformers"]
    factory, observed = module.AutoProcessor.from_pretrained, []
    def load(*args, **kwargs):
        observed.append(kwargs)
        return factory(*args, **kwargs)
    monkeypatch.setattr(module.AutoProcessor, "from_pretrained", load)
    before = build(materials, manifest=False)
    assert observed[-1]["use_fast"] is False
    assert observed[-1]["local_files_only"] is True
    monkeypatch.setattr(vlm_backend, "PROCESSOR_LOADING_SETTINGS", {"use_fast": True})
    after = build(materials, manifest=False)
    assert observed[-1]["use_fast"] is True
    assert before.identity.prompt_sha256 != after.identity.prompt_sha256


def test_terminal_forward_only_computes_last_token_head_and_keeps_abcd_alignment(materials):
    model, received = build(materials, manifest=False), []
    class Inputs(dict):
        def to(self, device):
            return self
    class Scores:
        def float(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return np.array([.11, .22, .33, .44])
    class Logits:
        def __getitem__(self, key):
            assert key == (0, -1, [0, 1, 2, 3])
            return Scores()
    class Forward:
        device = "fixture-cpu"
        def __call__(self, **kwargs):
            received.append(kwargs.get("logits_to_keep"))
            return SimpleNamespace(logits=Logits())
    model.processor.apply_chat_template = lambda *args, **kwargs: Inputs(input_ids=SimpleNamespace(shape=(1, 20)))
    model.model, model.torch = Forward(), SimpleNamespace(inference_mode=nullcontext)
    model.class_token_ids = [0, 1, 2, 3]
    values = model._call("terminal", materials.claim, {"claim": materials.claim.text}, CostLedger(), logits=True)
    assert received == [1]
    np.testing.assert_allclose(values, [.11, .22, .33, .44])


def test_terminal_head_loading_scope_is_part_of_frozen_identity(materials, monkeypatch):
    before = build(materials, manifest=False)
    monkeypatch.setattr(vlm_backend, "TERMINAL_LOGITS_SETTINGS", {"logits_to_keep": 0,
                       "score_position": "last_token", "label_token_order": "ABCD"})
    after = build(materials, manifest=False)
    assert before.identity.prompt_sha256 != after.identity.prompt_sha256
