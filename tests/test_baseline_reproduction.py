"""真实官方 FIRE 控制函数 + 显式合成依赖，不运行LLM/搜索服务。"""
import builtins
import json
from pathlib import Path

import pytest

from src import baseline_reproduction as baseline


class ScriptedRater:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []
    def generate(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(next(self.replies)), {"input_tokens": 10, "output_tokens": 3}


def controller(search=None, similarity=None):
    return baseline.OfficialFIREController(search or (lambda q: "Archived fixture evidence: "+q),
                similarity or (lambda new, old, threshold: sum(x == new for x in old)),
                rater_id="scripted-test-double", search_provider_id="explicit-fixture-search",
                similarity_provider_id="exact-match-test-double")


def test_real_official_controller_can_finalize_without_search():
    run = controller(search=lambda q: pytest.fail("search must not run")).verify("A fixture claim", ScriptedRater([{"final_answer": "True"}]))
    assert run["status"] == "completed" and run["decision"] == "True"
    assert run["search_trace"] == [] and run["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert run["provenance"]["upstream_commit"] == baseline.FIRE_COMMIT
    assert set(run["provenance"]["unchanged_control_function_ast_sha256"]) == set(baseline.CONTROL_FUNCTIONS)
    assert run["directly_comparable_to_four_class"] is False


def test_real_official_loop_searches_then_verifies():
    queries = []
    def search(query):
        queries.append(query)
        return "Actual archived fixture passage"
    rater = ScriptedRater([{"search_query": "source date"}, {"final_answer": "False"}])
    run = controller(search=search).verify("A fixture claim", rater)
    assert run["status"] == "completed" and run["decision"] == "False"
    assert queries == ["source date"] and len(rater.prompts) == 2
    assert "Actual archived fixture passage" in rater.prompts[1]
    assert run["usage"] == {"input_tokens": 20, "output_tokens": 6}


def test_real_official_duplicate_query_early_stop_calls_forced_finalizer():
    queries = []
    run = controller(search=lambda q: queries.append(q) or "fixture source").verify("A fixture claim",
        ScriptedRater([{"search_query": "same query"}, {"search_query": "same query"}, {"final_answer": "False"}]))
    assert run["status"] == "completed" and run["decision"] == "False"
    assert queries == ["same query"]
    assert len(run["rater_trace"]) == 3 and run["similarity_trace"][0]["count"] == 1


def test_real_official_step_limit_uses_native_forced_binary_answer():
    rater = ScriptedRater([{"search_query": "one query"}, {"final_answer": "True"}])
    run = controller().verify("A fixture claim", rater, max_steps=1)
    assert run["status"] == "completed" and len(rater.prompts) == 2
    assert "must" in rater.prompts[1].lower() or "final answer" in rater.prompts[1].lower()


def test_nonbinary_result_is_not_coerced_into_four_class_performance():
    run = controller().verify("A fixture claim", ScriptedRater([{"final_answer": "Not Enough Evidence"}]))
    assert run["status"] == "failed" and run["decision"] is None


def test_official_module_side_effect_dependencies_are_never_imported(monkeypatch):
    actual_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {"torch", "transformers", "sentence_transformers", "langchain_openai", "langchain_anthropic"}
        return actual_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    run = controller().verify("A fixture claim", ScriptedRater([{"final_answer": "True"}]))
    assert run["status"] == "completed"


def test_source_hash_and_reviewed_patch_are_enforced(monkeypatch, tmp_path):
    monkeypatch.setitem(baseline.SOURCE_HASHES, "common/utils.py", "0"*64)
    with pytest.raises(baseline.BaselineAdapterError, match="源码已变化"):
        controller()
    monkeypatch.undo()
    patch = tmp_path/"wrong.patch"
    patch.write_text("unreviewed modification", encoding="utf-8")
    monkeypatch.setattr(baseline, "PATCH", patch)
    with pytest.raises(baseline.BaselineAdapterError, match="patch"):
        controller()


def test_archived_search_validates_proof_and_missing_query_fails(tmp_path):
    body = "A real file used as a clearly synthetic test fixture."
    snapshot = tmp_path/"snapshot.txt"
    snapshot.write_text(body, encoding="utf-8")
    manifest = tmp_path/"searches.json"
    manifest.write_text(json.dumps({"queries": {"exact query": [{"url":"https://source.example.org/article",
        "snapshot_path":"snapshot.txt", "snapshot_sha256":baseline._sha(snapshot.read_bytes()),
        "start":0, "end":len(body), "text":body}]}}),encoding="utf-8")
    search = baseline.ArchivedFIRESearch(manifest)
    assert body in search("exact query")
    with pytest.raises(baseline.BaselineAdapterError, match="缺少真实归档"):
        search("missing query")
    run = controller(search=search).verify("A fixture claim", ScriptedRater([{"search_query":"missing query"}]))
    assert run["status"] == "failed" and run["decision"] is None
    snapshot.write_text("changed body", encoding="utf-8")
    with pytest.raises(baseline.BaselineAdapterError, match="SHA-256"):
        search("exact query")


def test_invalid_json_attempt_tokens_remain_in_actual_rater_accounting():
    class Rater:
        def __init__(self):
            self.replies = iter(["not JSON", '{"final_answer":"True"}'])
        def generate(self, _):
            return next(self.replies), {"input_tokens":10, "output_tokens":3}
    run = controller().verify("A fixture claim", Rater())
    assert run["status"] == "completed"
    # 忠实保留上游计数同时独立记录所有真实调用，不能因格式失败把实际token丢掉。
    assert run["usage"]["input_tokens"] == 10
    assert run["rater_recorded_usage"]["input_tokens"] == 20
    assert run["rater_recorded_usage"]["complete"] is True


def test_failed_search_attempt_is_visible_in_trace():
    def search(_):
        raise RuntimeError("explicit fixture failure")
    run = controller(search=search).verify("A fixture claim", ScriptedRater([{"search_query":"source"}]))
    assert run["status"] == "failed"
    assert run["search_trace"][0]["success"] is False
    assert "fixture failure" in run["search_trace"][0]["error"]


def test_failed_similarity_attempt_is_visible_in_trace():
    def similarity(*args, **kwargs):
        raise RuntimeError("explicit similarity fixture failure")
    run = controller(similarity=similarity).verify("A fixture claim",
        ScriptedRater([{"search_query":"same query"}, {"search_query":"same query"}]))
    assert run["status"] == "failed"
    assert run["similarity_trace"][0]["success"] is False
    assert "similarity fixture failure" in run["similarity_trace"][0]["error"]
