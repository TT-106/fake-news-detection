"""FIRE 本地依赖接口检查；Qwen 仅测试注入对象，真实 SBERT 需显式开启。"""
from dataclasses import dataclass
import json
import os

import pytest
import torch

from src.baseline_reproduction import BaselineAdapterError
from src.fire_local_dependencies import LocalSbertSimilarity, QwenFIREGeneration, QwenFIRERater


class Inputs(dict):
    def to(self, device):
        assert str(device) == "cpu"
        return self


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return Inputs(input_ids=torch.tensor([[1,2,3]]))
    def decode(self, tokens, **kwargs):
        assert len(tokens) == 2
        return '{"final_answer":"True"}'


class Model:
    device, training = "cpu", False
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("explicit fixture generation failure")
        return torch.tensor([[1,2,3,4,5]])


@dataclass(frozen=True)
class Identity:
    model_id: str = "EXPLICIT_TEST_DOUBLE"
    revision: str = "fixture-only"


class LoadedVLMFixture:
    def __init__(self, fail=False):
        self.model, self.processor, self.torch, self.identity = Model(fail), Processor(), torch, Identity()


def test_qwen_fire_rater_keeps_original_prompt_text_and_uses_text_list_messages(tmp_path):
    from src.vlm_backend import PROMPTS
    original_prompts = PROMPTS.copy()
    vlm = LoadedVLMFixture()
    original_identity = vlm.identity
    rater = QwenFIRERater(vlm,tmp_path)
    original_fire_prompt = 'Instructions:\nSTATEMENT: "2020 Paris"\nKNOWLEDGE: raw evidence'
    response, usage = rater.generate(original_fire_prompt)
    assert response == '{"final_answer":"True"}' and usage == {"input_tokens":3,"output_tokens":2}
    messages = vlm.processor.messages
    assert messages[0]["content"] == [{"type":"text","text":"You are a fact-checking agent responsible for verifying the accuracy of claims."}]
    assert messages[1]["content"] == [{"type":"text","text":original_fire_prompt}]
    assert vlm.model.calls[0]["do_sample"] is False
    assert vlm.identity is original_identity and PROMPTS == original_prompts
    record = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert record["prompt"] == original_fire_prompt and record["success"] and record["elapsed_ms"] >= 0
    assert record["rater_identity"] != vlm.identity.revision


def test_failed_qwen_generation_retains_input_usage_and_failure_log(tmp_path):
    rater = QwenFIRERater(LoadedVLMFixture(fail=True),tmp_path)
    with pytest.raises(RuntimeError,match="fixture generation failure"):
        rater.generate("Actual controller prompt fixture")
    record = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert record["success"] is False and record["input_tokens"] == 3 and record["output_tokens"] is None
    assert "fixture generation failure" in record["error"]


def test_qwen_length_bound_is_not_silently_accepted(tmp_path):
    rater = QwenFIRERater(LoadedVLMFixture(),tmp_path,max_new_tokens=2)
    with pytest.raises(BaselineAdapterError,match="截断"):
        rater.generate("controller prompt")
    record = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert record["success"] is False and record["output_tokens"] == 2 and record["response"]


def test_qwen_input_bound_fails_before_generation_and_keeps_prompt(tmp_path):
    vlm = LoadedVLMFixture()
    rater = QwenFIRERater(vlm,tmp_path,max_input_tokens=2)
    with pytest.raises(BaselineAdapterError,match="静默截断"):
        rater.generate("controller prompt")
    assert vlm.model.calls == []
    record = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert record["input_tokens"] == 3 and record["output_tokens"] is None


@pytest.mark.parametrize("changed",["identity","settings"])
def test_qwen_base_identity_or_generation_change_is_detected(tmp_path,changed):
    vlm = LoadedVLMFixture()
    rater = QwenFIRERater(vlm,tmp_path)
    if changed == "identity":
        vlm.identity = Identity(revision="changed")
    else:
        rater.settings = QwenFIREGeneration(7,4096)
    with pytest.raises(BaselineAdapterError,match="锁定后变化"):
        rater.generate("controller prompt")
    assert vlm.model.calls == []


@pytest.mark.local_model
@pytest.mark.skipif(os.environ.get("RUN_LOCAL_FIRE_SBERT") != "1",reason="显式启用真实本地CPU SBERT，绝不下载")
def test_actual_local_sbert_encoder_counts_real_cosine_similarity():
    similarity = LocalSbertSimilarity()
    sentence = "The Eiffel Tower is in Paris, France."
    assert similarity(sentence,[sentence,"A fish swims in the ocean."],threshold=.9) == 1
    assert similarity(sentence,[],threshold=.9) == 0
    assert similarity.calls[0]["cosine_scores"][0] > .99


@pytest.mark.local_model
@pytest.mark.skipif(os.environ.get("RUN_LOCAL_FIRE_SBERT") != "1",reason="显式启用真实本地CPU SBERT精准阈值检查，绝不下载")
def test_actual_sbert_threshold_equality_is_excluded_like_official_fire():
    similarity = LocalSbertSimilarity()
    new = "Dhoni attended a cricket match in England."
    history = ["Dhoni watched a cricket match in England."]
    similarity(new,history,threshold=0.)
    exact_cosine = similarity.calls[-1]["cosine_scores"][0]
    assert -1 < exact_cosine < 1
    assert similarity(new,history,threshold=exact_cosine) == 0
    assert similarity.calls[-1]["cosine_scores"][0] == exact_cosine
    assert "score>threshold" in similarity.identity_metadata["similarity"]
    print(json.dumps({"exact_cosine":exact_cosine,"threshold":exact_cosine,
                      "strict_count":similarity.calls[-1]["count"],"identity":similarity.identity}))
