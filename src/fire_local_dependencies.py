"""FIRE 的显式本地依赖：CPU SBERT 与共享已加载 Qwen 的文本 rater。

只从本地目录加载 SBERT，绝不下载。Qwen 必须由外部显式加载；本模块不创建
新模型、不突变全局提示词或底层模型身份。设备/解码差异均登记为 adapted。
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from src.baseline_reproduction import BaselineAdapterError, FIRE_COMMIT, ROOT


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class LocalSbertSimilarity:
    """真实官方 encoder + cosine 阈值计数；CPU 差异明确登记。"""
    def __init__(self, model_dir=ROOT/"models/all-MiniLM-L6-v2",
                 manifest_path=ROOT/"models/all-MiniLM-L6-v2.manifest.json"):
        self.directory = Path(model_dir).resolve()
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if (manifest.get("repository") != "sentence-transformers/all-MiniLM-L6-v2"
                or not isinstance(manifest.get("revision"), str) or len(manifest["revision"]) != 40
                or not isinstance(manifest.get("files"), list)):
            raise BaselineAdapterError("本地 SBERT 缺少锁定的官方模型清单")
        for item in manifest["files"]:
            path = (self.directory/item["path"]).resolve()
            if not path.is_relative_to(self.directory) or not path.is_file():
                raise BaselineAdapterError("SBERT 资源缺失或路径超出本地模型目录")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise BaselineAdapterError("SBERT 本地资源 SHA-256 不匹配："+item["path"])
        if importlib.metadata.version("sentence-transformers") != "3.1.1":
            raise BaselineAdapterError("FIRE 本地句向量依赖须为 sentence-transformers==3.1.1")
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(str(self.directory), device="cpu", local_files_only=True,
            trust_remote_code=False, model_kwargs={"local_files_only":True}, tokenizer_kwargs={"local_files_only":True})
        self.encoder.eval()
        self.identity_metadata = {"repository":manifest["repository"], "revision":manifest["revision"],
                                  "files_sha256":{r["path"]:r["sha256"] for r in manifest["files"]},
                                  "package":"sentence-transformers==3.1.1", "device":"cpu",
                                  "similarity":"sentence_transformers.util.cos_sim; score>threshold",
                                  "fidelity":"same encoder and cosine; CPU device adapted from original CUDA helper"}
        self.identity = "sbert-"+_digest(self.identity_metadata)
        self.calls = []

    def __call__(self, new_sentence, sentences, threshold=.9):
        if (not isinstance(new_sentence, str) or not isinstance(sentences, (list, tuple))
                or any(not isinstance(s, str) for s in sentences)
                or isinstance(threshold, bool) or not isinstance(threshold, (int,float))
                or not math.isfinite(threshold) or not -1 <= threshold <= 1):
            raise BaselineAdapterError("SBERT 文本列表或 cosine 阈值无效")
        started = time.perf_counter()
        if not sentences:
            scores = []
        else:
            from sentence_transformers import util
            # 与官方 helper 一样分别编码新句和历史列表；不使用词面近似代替真实 encoder。
            single = self.encoder.encode(new_sentence, convert_to_tensor=True, device="cpu")
            history = self.encoder.encode(list(sentences), convert_to_tensor=True, device="cpu")
            scores = [float(v) for v in util.cos_sim(single, history)[0].tolist()]
        count = sum(score > threshold for score in scores)
        self.calls.append({"new_sentence_sha256":_digest(new_sentence), "history_size":len(sentences),
                           "threshold":threshold, "cosine_scores":scores, "count":count,
                           "elapsed_ms":(time.perf_counter()-started)*1000, "device":"cpu"})
        return count


@dataclass(frozen=True)
class QwenFIREGeneration:
    max_new_tokens: int = 768
    max_input_tokens: int = 4096
    do_sample: bool = False


class QwenFIRERater:
    def __init__(self, loaded_vlm, record_dir, *, max_new_tokens=768, max_input_tokens=4096,
                 fire_repository=ROOT/"third_party/fire"):
        if (type(max_new_tokens) is not int or max_new_tokens < 1
                or type(max_input_tokens) is not int or max_input_tokens < 1):
            raise BaselineAdapterError("FIRE Qwen 输入/输出上限必须为正整数")
        for name in ("model", "processor", "torch", "identity"):
            if not hasattr(loaded_vlm, name):
                raise BaselineAdapterError("必须显式传入已经加载的 LocalQwenVLM，不能隐式创建模型")
        source_path = Path(fire_repository)/"common/modeling.py"
        source = source_path.read_text(encoding="utf-8")
        values = [ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id == "SYS_PROMPT" for t in n.targets)]
        official = "You are a fact-checking agent responsible for verifying the accuracy of claims."
        if values != [official]:
            raise BaselineAdapterError("FIRE 官方系统提示发生变化，须重新核查适配")
        self.vlm, self.model, self.processor = loaded_vlm, loaded_vlm.model, loaded_vlm.processor
        self.settings = QwenFIREGeneration(max_new_tokens,max_input_tokens)
        self.system_prompt = official
        self._base_identity = asdict(loaded_vlm.identity) if is_dataclass(loaded_vlm.identity) else loaded_vlm.identity
        self._base_identity_digest = _digest(self._base_identity)
        self.identity_metadata = {"base_model_identity":self._base_identity,"fire_upstream_commit":FIRE_COMMIT,
            "fire_modeling_source_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "system_prompt":official,"generation":asdict(self.settings),
            "message_format":"system and user both content=[{type:text,text:...}]; no image input",
            "device":str(self.model.device),"fidelity":"local deterministic Qwen text-rater adapted; not original paper reproduction"}
        self.identity = "qwen-fire-rater-"+_digest(self.identity_metadata)
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True,exist_ok=True)
        self.records = []

    def generate(self, prompt):
        request_id = uuid4().hex
        started = time.perf_counter()
        response, error, input_tokens, output_tokens, success = None, None, None, None, False
        messages = [{"role":"system","content":[{"type":"text","text":self.system_prompt}]},
                    {"role":"user","content":[{"type":"text","text":prompt}]}]
        try:
            current_identity = asdict(self.vlm.identity) if is_dataclass(self.vlm.identity) else self.vlm.identity
            if (_digest(current_identity) != self._base_identity_digest or self.vlm.model is not self.model
                    or self.vlm.processor is not self.processor or self.system_prompt != self.identity_metadata["system_prompt"]
                    or asdict(self.settings) != self.identity_metadata["generation"]):
                raise BaselineAdapterError("共享模型、系统提示或FIRE生成配置在锁定后变化")
            if not isinstance(prompt,str) or not prompt.strip():
                raise BaselineAdapterError("FIRE原始提示须为非空文本")
            if getattr(self.model,"training",False):
                raise BaselineAdapterError("FIRE rater 只能使用已冻结eval模型")
            inputs = self.processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,
                         return_dict=True,return_tensors="pt").to(self.model.device)
            input_tokens = int(inputs["input_ids"].shape[-1])
            if input_tokens > self.settings.max_input_tokens:
                raise BaselineAdapterError("FIRE输入超过预先锁定上限，不能静默截断原始控制器提示")
            with self.vlm.torch.inference_mode():
                generated = self.model.generate(**inputs,do_sample=False,max_new_tokens=self.settings.max_new_tokens)
            output_tokens = int(generated.shape[-1]-input_tokens)
            response = self.processor.decode(generated[0,input_tokens:],skip_special_tokens=True).strip()
            if output_tokens >= self.settings.max_new_tokens:
                raise BaselineAdapterError("FIRE生成达到预先锁定长度上限，拒绝将截断输出当作有效回复")
            success = True
            return response,{"input_tokens":input_tokens,"output_tokens":output_tokens}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record = {"request_id":request_id,"rater_identity":self.identity,"base_model_identity":self._base_identity,
                "system_prompt":self.system_prompt,"prompt":prompt,"messages":messages,"response":response,
                "generation":asdict(self.settings),"input_tokens":input_tokens,"output_tokens":output_tokens,
                "elapsed_ms":(time.perf_counter()-started)*1000,"success":success,"error":error,
                "billing":"LOCAL_INFERENCE_NO_PAID_API; hardware/electricity not inferred"}
            target = self.record_dir/f"{request_id}.json"
            target.write_text(json.dumps(record,ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8")
            self.records.append(str(target))
