"""FIRE 官方二分类控制器的外置依赖注入；不导入上游有副作用的模块。

官方三个控制函数及提示词直接从锁定源码读入，函数体不修改。显式提供模型、
检索与相似度回调，移除无关 import/全局模型初始化的差异单独保存在审阅 patch。
不创建假第三方模块，不写 sys.modules，不下载模型，也不提供收费 API 实现。
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Callable
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
FIRE_COMMIT = "cd7cb4af8dab00b1ab4bd78197a52e052203873d"
SOURCE_HASHES = {
    "eval/fire/verify_atomic_claim.py": "00824127645acebb1b8df3d5d55604c084ea76a888c123e89311048c21cd0e58",
    "eval/fire/config.py": "77209ce1459e240f3b00231df21738ff75e76abf84c9708c011698e4b63ef71d",
    "common/utils.py": "dc5d7b7bf5609fbe85223144169805a64d163874cf30613f733b72b54249f6a8",
}
CONTROL_FUNCTIONS = ("final_answer_or_next_search", "must_get_final_answer", "verify_atomic_claim")
PATCH = ROOT/"docs/baselines/fire_dependency_injection.patch"


class BaselineAdapterError(ValueError):
    pass


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _segment(source, node):
    first = min([node.lineno]+[d.lineno for d in getattr(node, "decorator_list", [])])
    return "".join(source.splitlines(keepends=True)[first-1:node.end_lineno]).rstrip()


def _projection(source):
    names = {"GoogleSearchResult", "FinalAnswer", *CONTROL_FUNCTIONS}
    constants = {"_Factual_LABEL", "_Non_Factual_LABEL", "_STATEMENT_PLACEHOLDER", "_KNOWLEDGE_PLACEHOLDER",
                 "_FINAL_ANSWER_OR_NEXT_SEARCH_FORMAT", "_MUST_HAVE_FINAL_ANSWER_FORMAT"}
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            selected.append(_segment(source, node))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            selected.append(_segment(source, node))
    return "from __future__ import annotations\nimport dataclasses\nfrom typing import Any\n\n"+"\n\n".join(selected)+"\n"


def fire_review_patch(source):
    """完整可审阅差异；patch 内容与实际投影代码逐字匹配。"""
    return "".join(difflib.unified_diff(source.splitlines(keepends=True), _projection(source).splitlines(keepends=True),
                                       fromfile="a/eval/fire/verify_atomic_claim.py", tofile="b/fire_controller.dependencies_injected.py"))


@dataclass(frozen=True)
class _Configuration:
    max_steps: int
    max_retries: int
    max_tolerance: int
    diverse_prompt: bool


@dataclass(frozen=True)
class _UtilityBindings:
    strip_string: Callable
    extract_json_from_output: Callable


def _load_official(repo, search, similarity_count):
    repo = Path(repo).resolve()
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != FIRE_COMMIT:
        raise BaselineAdapterError("FIRE 提交不匹配；必须先重新审查与锁定适配 patch")
    texts = {}
    for name, expected in SOURCE_HASHES.items():
        raw = (repo/name).read_bytes()
        if _sha(raw) != expected:
            raise BaselineAdapterError(f"FIRE 上游源码已变化：{name}")
        texts[name] = raw.decode("utf-8").replace("\r\n", "\n")
    source = texts["eval/fire/verify_atomic_claim.py"]
    patch = fire_review_patch(source)
    if not PATCH.is_file() or PATCH.read_text(encoding="utf-8") != patch:
        raise BaselineAdapterError("FIRE 已审查 patch 缺失或与实际依赖注入差异不一致")
    defaults = {}
    for node in ast.parse(texts["eval/fire/config.py"]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"max_steps", "max_retries", "max_tolerance", "diverse_prompt"}:
                defaults[node.targets[0].id] = ast.literal_eval(node.value)
    helpers = {"re": re, "json": json}
    helper_nodes = {n.name: n for n in ast.parse(texts["common/utils.py"]).body if isinstance(n, ast.FunctionDef)}
    helper_source = "\n\n".join(_segment(texts["common/utils.py"], helper_nodes[n]) for n in
                                 ("strip_string", "extract_json_from_output"))
    exec(compile(helper_source, "<official_fire_pure_helpers>", "exec"), helpers)
    bindings = _UtilityBindings(helpers["strip_string"], helpers["extract_json_from_output"])
    projected = _projection(source)
    original_tree, projected_tree = ast.parse(source), ast.parse(projected)
    function_hashes = {}
    for name in CONTROL_FUNCTIONS:
        original = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        loaded = next(n for n in projected_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if ast.dump(original, include_attributes=False) != ast.dump(loaded, include_attributes=False):
            raise BaselineAdapterError(f"FIRE 官方控制函数被改写：{name}")
        function_hashes[name] = _sha(ast.dump(original, include_attributes=False).encode("utf-8"))
    # 这是明确的依赖注入绑定，均为普通对象/回调；没有伪造或遮蔽第三方 import 模块。
    namespace = {"__name__": __name__, "fire_config": _Configuration(**defaults), "utils": bindings,
                 "call_search": search, "get_sentence_similarity": similarity_count}
    exec(compile(projected, "<official_fire_control_dependencies_injected>", "exec"), namespace)
    provenance = {"upstream_repository": "https://github.com/mbzuai-nlp/fire", "upstream_commit": head,
                  "upstream_source_sha256": SOURCE_HASHES.copy(), "dependency_patch": str(PATCH.relative_to(ROOT)),
                  "dependency_patch_sha256": _sha(patch.encode("utf-8")), "projected_source_sha256": _sha(projected.encode("utf-8")),
                  "unchanged_control_function_ast_sha256": function_hashes,
                  "fidelity": "ORIGINAL_CONTROLLER_FUNCTIONS_UNCHANGED_DEPENDENCIES_INJECTED"}
    return namespace["verify_atomic_claim"], provenance


class ArchivedFIRESearch:
    """按精确查询回放已归档片段；缺查询或快照损坏就失败，绝不从金标准QA补全。

    清单格式：{"queries": {"exact query": [{"url", "snapshot_path", "snapshot_sha256",
    "start", "end", "text"}]}}。路径相对清单目录，或使用已归档绝对路径。
    此层验证正文，不验证原始账单；不能据此宣称同费用预算复现。
    """
    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        self.manifest_sha256 = _sha(self.path.read_bytes())
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(self.data.get("queries"), dict):
            raise BaselineAdapterError("FIRE 检索回放清单必须按精确查询归档")
        self.receipts = []

    def __call__(self, query):
        if _sha(self.path.read_bytes()) != self.manifest_sha256:
            raise BaselineAdapterError("FIRE 检索回放清单在锁定后发生变化")
        if query not in self.data["queries"]:
            raise BaselineAdapterError(f"该查询缺少真实归档返回：{query}")
        passages = self.data["queries"][query]
        if not isinstance(passages, list):
            raise BaselineAdapterError("归档查询返回必须为片段列表，成功空返回须显式 []")
        texts, proofs = [], []
        for row in passages:
            path = Path(row["snapshot_path"])
            if not path.is_absolute():
                path = self.path.parent/path
            raw = path.read_bytes()
            if _sha(raw) != row["snapshot_sha256"]:
                raise BaselineAdapterError("FIRE 检索快照 SHA-256 不匹配")
            body = raw.decode("utf-8")
            start, end = row["start"], row["end"]
            url = urlsplit(row["url"])
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(body)
                    or body[start:end] != row["text"] or url.scheme not in {"http", "https"} or not url.hostname):
                raise BaselineAdapterError("FIRE 检索引文位置、正文或来源URL无效")
            texts.append(row["url"]+"\n"+row["text"])
            proofs.append({k: row[k] for k in ("url", "snapshot_sha256", "start", "end")})
        self.receipts.append({"query": query, "manifest_sha256": self.manifest_sha256, "passages": proofs,
                              "billing_status": "NOT_VALIDATED_IN_THIS_ADAPTER"})
        return "\n\n".join(texts)


class OfficialFIREController:
    name = "FIRE original-controller local-rater adapted"

    def __init__(self, search: Callable, similarity_count: Callable, *,
                 rater_id: str, search_provider_id: str, similarity_provider_id: str,
                 repo=ROOT/"third_party/fire"):
        if not callable(search) or not callable(similarity_count):
            raise BaselineAdapterError("必须显式注入搜索及相似查询计数回调；没有默认网络/模型下载路径")
        if not all(isinstance(v, str) and v.strip() for v in (rater_id, search_provider_id, similarity_provider_id)):
            raise BaselineAdapterError("必须记录实际模型、搜索与相似度提供者身份")
        self.search_calls, self.similarity_calls = [], []
        def call_search(query):
            if not isinstance(query, str) or not query.strip():
                raise BaselineAdapterError("FIRE 查询必须为非空文本")
            trace = {"query": query, "success": False}
            self.search_calls.append(trace)
            try:
                result = search(query)
            except Exception as exc:
                trace["error"] = f"{type(exc).__name__}: {exc}"
                raise
            if not isinstance(result, str):
                raise BaselineAdapterError("FIRE 搜索回调必须返回归档正文文本")
            trace.update(success=True, response_sha256=_sha(result.encode("utf-8")))
            return result
        def count_similar(new, sentences, threshold=.9):
            trace = {"sentences": len(sentences), "threshold": threshold, "success": False}
            self.similarity_calls.append(trace)
            try:
                value = similarity_count(new, sentences, threshold=threshold)
            except Exception as exc:
                trace["error"] = f"{type(exc).__name__}: {exc}"
                raise
            if type(value) is not int or not 0 <= value <= len(sentences):
                raise BaselineAdapterError("FIRE 相似度回调必须返回满足阈值的句子数量")
            trace.update(success=True, count=value)
            return value
        self.controller, self.provenance = _load_official(repo, call_search, count_similar)
        self.provenance.update(rater_id=rater_id, search_provider_id=search_provider_id,
                               similarity_provider_id=similarity_provider_id)

    def verify(self, claim_text, rater, *, max_steps=5, max_retries=10, tolerance=2, diverse_prompt=False):
        if not isinstance(claim_text, str) or not claim_text.strip() or not callable(getattr(rater, "generate", None)):
            raise BaselineAdapterError("FIRE 需要原始声明文本与显式 rater.generate 接口")
        if (type(max_steps) is not int or max_steps < 1 or type(max_retries) is not int or max_retries < 0
                or type(tolerance) is not int or tolerance < 2 or type(diverse_prompt) is not bool):
            raise BaselineAdapterError("FIRE 步数、重试、重复容忍度或提示开关无效")
        start_search, start_similarity = len(self.search_calls), len(self.similarity_calls)
        traces = []
        class RecordedRater:
            def generate(self, prompt):
                trace = {"prompt_sha256": _sha(prompt.encode("utf-8")), "success": False}
                traces.append(trace)
                try:
                    response, usage = rater.generate(prompt)
                    if not isinstance(response, str):
                        raise BaselineAdapterError("本地 rater 必须返回文本，不能把结构化判定直接塞给控制器")
                    if usage is not None and (not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens"))):
                        raise BaselineAdapterError("rater usage 必须提供真实非负 input_tokens/output_tokens，或明确为None")
                except Exception as exc:
                    trace["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                trace.update(success=True, response=response, usage=usage)
                return response, usage
        result, searches, usage, error = None, {}, None, None
        try:
            result, searches, usage = self.controller(claim_text, RecordedRater(), max_steps=max_steps,
                                                       max_retries=max_retries, tolerance=tolerance, diverse_prompt=diverse_prompt)
            if result is None or result.answer not in {"True", "False"}:
                error = "官方控制器未产生合法原生二分类判定"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        measured = [t["usage"] for t in traces if t.get("usage") is not None]
        measured_usage = {key: sum(u[key] for u in measured) for key in ("input_tokens", "output_tokens")}
        measured_usage.update(complete=len(measured) == len(traces), measured_calls=len(measured), attempted_calls=len(traces))
        return {"name": self.name, "status": "failed" if error else "completed", "error": error,
                "decision": result.answer if result is not None and not error else None,
                "task": "native_atomic_claim_binary", "native_labels": ["True", "False"],
                "directly_comparable_to_four_class": False,
                "scope": "控制器运行记录；不自动证明论文复现、来源充分性或同费用预算验收",
                "searches": searches, "usage": usage, "usage_scope": "ORIGINAL_CONTROLLER_COUNTER_MAY_OMIT_INVALID_RESPONSE_ATTEMPTS",
                "rater_recorded_usage": measured_usage, "rater_trace": traces,
                "search_trace": self.search_calls[start_search:], "similarity_trace": self.similarity_calls[start_similarity:],
                "controller_parameters": {"max_steps": max_steps, "max_retries": max_retries,
                                          "tolerance": tolerance, "diverse_prompt": diverse_prompt},
                "provenance": self.provenance.copy()}
