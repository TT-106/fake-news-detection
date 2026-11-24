"""本地冻结 Qwen2.5-VL：仅从已有目录加载，永不下载或静默改用 NLI。

立场是 VLM 对每条原子声明独立给出的双通道评分；终端四类原始得分
取 A/B/C/D 单 token 的真实 forward logits，不使用模型自报的分类概率。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from src.core_models import LABELS, SoftLogicRule
from src.faithful_pipeline import ClaimAnalysis, ModelIdentity
from src.input_views import InputView
from src.learned_components import ReviewedLabel
from src.research_contracts import (ArchivedPassage, AtomicClaim, CanonicalFact, CostEntry, CostLedger,
                                    ProtocolError, ScoredPassage, ScopedRule, TerminalOutput, fingerprint)


PROMPTS = {
    "analyse": """Read the original images and claim. Decompose the claim into 1 to 4 independently
verifiable atomic claims. Separate visible image content from the historical event attribution.
Use only the given claim, images and optional preverified logical-relation catalogue.
The catalogue is NOT a verdict or retrieved evidence. Never infer that a caption is true from it.
Return JSON only:
{"atoms":[{"atom_id":"a0","text":"...","entity":"...","event":"...","time":"","location":""}],
"gaps":["specific fact or provenance questions still requiring evidence"],
"rules":[{"rule_id":"an existing catalogue ID","antecedent_atom_id":"a0","consequent_atom_id":"a1"}]}.
Select a rule only when both atom texts and entity/event/time/location EXACTLY match its catalogue
endpoints. Preserve those texts when selecting. implication means antecedent implies consequent;
necessary means the consequent is necessary for the antecedent; incompatibility means the two
distinct propositions cannot both hold in that exact context. No catalogue means rules=[].
Do not invent coefficients, relation IDs, evidence, reliability scores or generic rules.
Do not make support and refutation universally exclusive: credible conflict must remain representable.""",
    "score": """Read the original images, caption, atomic claims and the retrieved passage.
Return JSON only with fact (entity,event,relation,polarity,time,location), r_pos, r_neg,
and quality_features. Each score array has one entry per atomic claim in the given order.
The support and refutation ratings are INDEPENDENT numbers in [0,1], not a softmax;
both may be low for missing evidence, or high for conflicting credible evidence.
quality_features is one row per atom: [traceability, passage completeness, applicability],
each in [0,1]. Quality means evidence usefulness, NOT whether it supports the caption.
A dated original archive refuting a current caption may be high-quality evidence.
The canonical fact must include actual entity, event, relation, polarity and time/place qualifiers;
never use 'time', 'place', or a question slot as a fact identifier. Do not invent sources.""",
    "terminal": """Read the original images, caption, actual retrieved passages, support/refutation
states and unresolved gaps. Classify the claim using EXACTLY one character:
A = Supported; B = Refuted; C = Not Enough Evidence; D = Conflicting Evidence/Cherrypicking.
Evidence conflict differs from evidence absence. Do not classify from topic similarity alone.""",
    "citations": """Read the proposed label and existing evidence. Return JSON only:
{"citation_ids":["existing evidence IDs actually supporting the proposed label"],"rationale":"..."}.
Use an empty list if no available passage supports the label. Never invent an evidence ID.""",
}


# RTX 4060 8GB 的显式加载口径；不是通过评测结果搜索出来的性能超参数。
# dtype 用稳定字符串进入身份；加载时再映射到真实 torch dtype。
BNB4_LOADING_SETTINGS = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}
PROCESSOR_LOADING_SETTINGS = {"use_fast": False}
TERMINAL_LOGITS_SETTINGS = {"logits_to_keep": 1, "score_position": "last_token", "label_token_order": "ABCD"}


def public_claim_digest(claim) -> str:
    """可移植地绑定事前公开输入，不纳入真值、QA、文件路径或事后解释。"""
    return fingerprint({"claim_id": claim.claim_id, "text": claim.text, "date": claim.date,
                        "location": claim.location,
                        "image_sha256": sorted(asset.sha256 for asset in claim.images)})


def _model_resource_hashes(directory: Path) -> dict[str, str]:
    """冻结整个本地模型包，包括嵌套模板/词表；不猜处理器可能读取的后缀。"""
    hashes = {}
    def visit(folder, ancestors):
        resolved = folder.resolve()
        if resolved in ancestors:
            raise ProtocolError("模型资源目录含循环链接，不能可靠冻结")
        for resource in sorted(folder.iterdir()):
            if resource.is_dir():
                visit(resource, ancestors | {resolved})
            elif resource.is_file():
                with resource.open("rb") as stream:
                    hashes[resource.relative_to(directory).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
            else:
                raise ProtocolError("模型包存在无法读取的资源，不能省略后继续冻结")
    visit(directory, set())
    return hashes


@dataclass(frozen=True)
class _ReviewedRelation:
    rule_id: str
    claim_id: str
    input_sha256: str
    kind: str
    antecedent: tuple[str, ...]  # text, entity, event, time, location
    consequent: tuple[str, ...]
    weight: float
    justification: str
    proofs: tuple[ArchivedPassage, ...]

    def visible(self):
        names = ("text", "entity", "event", "time", "location")
        return {"rule_id": self.rule_id, "rule_type": self.kind,
                "antecedent": dict(zip(names, self.antecedent)),
                "consequent": dict(zip(names, self.consequent))}


def _load_rule_catalogue(path: str | None):
    if path is None:
        return (), fingerprint({"protocol": "NO_PREVERIFIED_RULE_CATALOGUE"})
    manifest = Path(path).resolve()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if (data.get("protocol") != "PREVERIFIED_SCOPED_RULES_V1"
                or data.get("supervision_scope") != "logical_relation_only"
                or not isinstance(data.get("records"), list)):
            raise ProtocolError("规则材料必须声明预核验逻辑关系协议，不得使用判定标签/QA 生成规则")
        records, seen = [], set()
        required = {"rule_id", "claim_id", "claim_input_sha256", "rule_type", "antecedent", "consequent",
                    "weight", "justification", "review", "proofs"}
        names = ("text", "entity", "event", "time", "location")
        for row in data["records"]:
            if not isinstance(row, dict) or set(row) != required:
                raise ProtocolError("规则记录字段不符合预核验协议；不接受任意系数或隐藏真值")
            if (not isinstance(row["rule_id"], str) or not row["rule_id"].strip()
                    or not isinstance(row["claim_id"], str) or not row["claim_id"].strip()
                    or row["rule_id"] in seen):
                raise ProtocolError("规则 ID/声明 ID 缺失或规则 ID 重复")
            seen.add(row["rule_id"])
            digest = row["claim_input_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ProtocolError("规则必须绑定公开声明输入的 SHA-256")
            endpoints = []
            for key in ("antecedent", "consequent"):
                endpoint = row[key]
                if (not isinstance(endpoint, dict) or set(endpoint) != set(names)
                        or any(not isinstance(endpoint[n], str) or not endpoint[n].strip() for n in names)):
                    raise ProtocolError("可靠规则须有非空原子文本和明确实体/事件/时间/地点，不能扩大适用域")
                endpoints.append(tuple(endpoint[n] for n in names))
            if endpoints[0][1:] != endpoints[1][1:] or endpoints[0][0] == endpoints[1][0]:
                raise ProtocolError("规则必须连接同一明确语境中的两个不同命题")
            if row["rule_type"] not in {"implication", "necessary", "incompatibility"}:
                raise ProtocolError("未知规则关系类型")
            weight = row["weight"]
            if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight <= 0:
                raise ProtocolError("已核验规则权重必须是正有限数，不能使用负权重或无效占位")
            if not isinstance(row["justification"], str) or not row["justification"].strip():
                raise ProtocolError("可靠规则缺少逻辑关系依据")
            if ReviewedLabel(**row["review"]).resolve() != 1:
                raise ProtocolError("规则须经双人独立复核/必要仲裁确认，VLM 自报可靠不构成核验")
            if not isinstance(row["proofs"], list) or not row["proofs"]:
                raise ProtocolError("规则缺少真实可回放的依据材料")
            proofs = []
            for proof in row["proofs"]:
                proof = dict(proof)
                snapshot = Path(proof["snapshot_path"])
                proof["snapshot_path"] = str(snapshot if snapshot.is_absolute() else manifest.parent/snapshot)
                passage = ArchivedPassage(**proof)
                passage.verify()
                proofs.append(passage)
            records.append(_ReviewedRelation(row["rule_id"], row["claim_id"], digest, row["rule_type"],
                                             *endpoints, float(weight), row["justification"], tuple(proofs)))
        return tuple(records), fingerprint(data)
    except ProtocolError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ProtocolError(f"规则材料无法校验：{exc}") from exc


class LocalQwenVLM:
    def __init__(self, model_dir: str, quantization: str, record_dir: str,
                 max_pixels: int = 512*28*28, max_new_tokens: int = 768,
                 rule_manifest_path: str | None = None, extra_prompts: dict[str, str] | None = None):
        directory = Path(model_dir).resolve()
        if not directory.is_dir() or not (directory/"config.json").is_file():
            raise ProtocolError("本地 Qwen 模型未就绪；未下载模型，也不会改用替代打分器")
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        text_config = config.get("text_config") or config
        if (config.get("model_type") != "qwen2_5_vl" or text_config.get("hidden_size") != 3584
                or text_config.get("num_hidden_layers") != 28):
            raise ProtocolError("当前适配器只支持明确的 Qwen2.5-VL 版本；不能混用其他模型")
        weights = sorted(directory.glob("*.safetensors"))
        if not weights:
            raise ProtocolError("本地缺少实际 Qwen 权重")
        if quantization not in {"bf16", "bnb4", "awq4"} or max_pixels <= 0 or max_new_tokens <= 0:
            raise ProtocolError("无效的量化/图像分辨率/输出上限设置")
        if Path(record_dir).resolve().is_relative_to(directory):
            raise ProtocolError("运行记录目录不能位于冻结模型资源目录内")
        file_hashes = _model_resource_hashes(directory)
        self._rule_catalogue, self._rule_catalogue_digest = _load_rule_catalogue(rule_manifest_path)
        self.quantization_settings = dict(BNB4_LOADING_SETTINGS) if quantization == "bnb4" else {}
        self.processor_settings = dict(PROCESSOR_LOADING_SETTINGS)
        self.terminal_logits_settings = dict(TERMINAL_LOGITS_SETTINGS)
        self.prompts = {**PROMPTS, **(extra_prompts or {})}
        self.identity = ModelIdentity("Qwen2.5-VL-7B-Instruct", fingerprint(file_hashes),
                                      fingerprint({"prompts": self.prompts, "max_pixels": max_pixels,
                                                   "max_new_tokens": max_new_tokens,
                                                   "rule_catalogue": self._rule_catalogue_digest,
                                                   "settings": {"quantization": self.quantization_settings,
                                                                "processor": self.processor_settings,
                                                                "terminal_logits": self.terminal_logits_settings,
                                                                "chat_content_format": "MULTIMODAL_CONTENT_BLOCKS_V1"}}), quantization)
        self.record_dir = Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.view = InputView()
        self.view_version = self.view.version
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        if not torch.cuda.is_available():
            raise ProtocolError("当前 PyTorch 无可用 CUDA；未尝试下载或替换运行环境")
        kwargs = dict(local_files_only=True, device_map="auto", dtype=torch.bfloat16)
        if quantization == "bnb4":
            from transformers import BitsAndBytesConfig
            quantization_kwargs = dict(self.quantization_settings)
            quantization_kwargs["bnb_4bit_compute_dtype"] = getattr(torch, quantization_kwargs["bnb_4bit_compute_dtype"])
            kwargs["quantization_config"] = BitsAndBytesConfig(**quantization_kwargs)
        if quantization == "awq4" and (config.get("quantization_config") or {}).get("quant_method") != "awq":
            raise ProtocolError("awq4 必须对应实际 AWQ 权重，不能只修改实验名称")
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True,
                                                       min_pixels=256*28*28, max_pixels=max_pixels,
                                                       **self.processor_settings)
        self.model = AutoModelForImageTextToText.from_pretrained(directory, **kwargs).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.torch = torch
        tokens = [self.processor.tokenizer.encode(x, add_special_tokens=False) for x in "ABCD"]
        if any(len(t) != 1 for t in tokens):
            raise ProtocolError("终端标签不是单 token，需要显式实现标签序列打分，不能伪造 logits")
        self.class_token_ids = [t[0] for t in tokens]

    def _call(self, operation, claim, payload, ledger: CostLedger, logits=False, *, schema=None):
        from PIL import Image
        image_objects = []
        try:
            for asset in (claim.images if self.view.include_images else ()):
                asset.verify()
                with Image.open(asset.path) as source:
                    rgb = source.convert("RGB")
                    try:
                        image_objects.append(self.view.project_image(asset.sha256, rgb))
                    finally:
                        rgb.close()
            content = [{"type": "image", "image": im} for im in image_objects]
            visible_payload = self.view.render(claim.claim_id, payload)
            content.append({"type": "text", "text": json.dumps(visible_payload, ensure_ascii=False)})
            if schema is not None:
                instruction = self.prompts[operation] + "\nReturn a JSON object matching this schema:\n" + json.dumps(schema, separators=(",", ":"))
                content.append({"type":"text", "text":instruction})
            messages = [{"role": "system", "content": [{"type": "text", "text": getattr(self, "prompts", PROMPTS)[operation]}]},
                        {"role": "user", "content": content}]
            request = {"operation": operation, "model": asdict(self.identity),
                       "claim": asdict(claim), "payload": visible_payload, "input_view": self.view_version}
            generation = {}
            if schema is not None:
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
                from lmformatenforcer.integrations.transformers import (build_token_enforcer_tokenizer_data,
                                                                      build_transformers_prefix_allowed_tokens_fn)
                if not hasattr(self, "_schema_tokenizer_data"):
                    self._schema_tokenizer_data = build_token_enforcer_tokenizer_data(self.processor.tokenizer)
                grammar = JsonSchemaParser(schema, config=CharacterLevelParserConfig(
                    force_json_field_order=True, max_consecutive_whitespaces=2))
                generation["prefix_allowed_tokens_fn"] = build_transformers_prefix_allowed_tokens_fn(self._schema_tokenizer_data, grammar)
                request["output_schema"] = schema
            request_id, request_hash = uuid.uuid4().hex, fingerprint(request)
            started, success, response = time.perf_counter(), False, None
            input_tokens, output_tokens = 0, 0
            try:
                inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                             return_dict=True, return_tensors="pt").to(self.model.device)
                input_tokens = int(inputs["input_ids"].shape[-1])
                with self.torch.inference_mode():
                    if logits:
                        values = self.model(**inputs, logits_to_keep=self.terminal_logits_settings["logits_to_keep"]).logits[
                            0, -1, self.class_token_ids].float().cpu().numpy()
                        response = [float(x) for x in values]
                    else:
                        generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens, **generation)
                        output_tokens = int(generated.shape[-1]-input_tokens)
                        text = self.processor.decode(generated[0, input_tokens:], skip_special_tokens=True).strip()
                        response = text
                        if output_tokens >= self.max_new_tokens:
                            raise ProtocolError("VLM 输出达到长度上限，拒绝把截断 JSON 当作有效结果")
                        if text.startswith("```json") and text.endswith("```"):
                            text = text[7:-3].strip()
                        response = text
                        response = json.loads(text)
                success = True
                return response
            finally:
                entry = CostEntry(request_id, f"qwen.{operation}", 0.,
                                  (time.perf_counter()-started)*1000, success, request_hash,
                                  "本地推理无 API 账单；记录耗时，未把电费/硬件摊销虚构成人民币费用",
                                  input_tokens, output_tokens)
                ledger.add(entry)
                record = {"request": request, "response": response, "cost": asdict(entry)}
                (self.record_dir/f"{request_id}.json").write_text(json.dumps(record, ensure_ascii=False,
                                                                           allow_nan=False, indent=2), encoding="utf-8")
        finally:
            for image in image_objects:
                image.close()

    def analyse(self, claim, ledger):
        audit = {"claim_id": claim.claim_id, "rule_catalogue_sha256": self._rule_catalogue_digest,
                 "model_revision": self.identity.revision}
        try:
            available = {r.rule_id: r for r in self._rule_catalogue if r.claim_id == claim.claim_id}
            if any(r.input_sha256 != public_claim_digest(claim) for r in available.values()):
                raise ProtocolError("规则材料绑定的公开声明版本不匹配")
            for relation in available.values():
                for proof in relation.proofs:
                    proof.verify()
            data = self._call("analyse", claim, {"claim": claim.text, "date": claim.date,
                                                "location": claim.location,
                                                "rule_catalogue": [r.visible() for r in available.values()]}, ledger)
            if not isinstance(data, dict) or set(data) - {"atoms", "gaps", "rules"}:
                raise ProtocolError("VLM 分析字段无效；规则不能通过未声明字段被静默丢弃")
            atoms = tuple(AtomicClaim(**a) for a in data["atoms"])
            if not 1 <= len(atoms) <= 4 or len({a.atom_id for a in atoms}) != len(atoms):
                raise ProtocolError("VLM 原子声明数量/唯一身份不符合申请书")
            gaps = data["gaps"]
            if not isinstance(gaps, list) or any(not isinstance(g, str) or not g.strip() for g in gaps):
                raise ProtocolError("事实缺口必须是明确非空文本列表")
            proposals = data.get("rules", [])
            if not isinstance(proposals, list):
                raise ProtocolError("VLM 规则选择必须是列表")
            atom_map = {a.atom_id: (i, a) for i, a in enumerate(atoms)}
            rules, selected = [], set()
            for proposal in proposals:
                if (not isinstance(proposal, dict)
                        or set(proposal) != {"rule_id", "antecedent_atom_id", "consequent_atom_id"}
                        or not all(isinstance(v, str) for v in proposal.values())):
                    raise ProtocolError("规则只能选择已核验 ID 与两个原子 ID，不接受模型任意系数/可靠性自评")
                rid = proposal["rule_id"]
                if rid not in available or rid in selected:
                    raise ProtocolError("规则 ID 未经核验、缺少规则材料或重复选择")
                selected.add(rid)
                relation = available[rid]
                endpoints = []
                for key, expected in (("antecedent_atom_id", relation.antecedent),
                                      ("consequent_atom_id", relation.consequent)):
                    if proposal[key] not in atom_map:
                        raise ProtocolError("规则选择了不存在的原子声明")
                    index, atom = atom_map[proposal[key]]
                    actual = (atom.text, atom.entity, atom.event, atom.time, atom.location)
                    if actual != expected:
                        raise ProtocolError("规则端点文本/实体/事件/时空不符合已核验适用域")
                    endpoints.append(index)
                i, j = endpoints
                if i == j:
                    raise ProtocolError("不得为单个原子声明添加通用正反互斥或自指规则")
                c, d, m = np.zeros(2*len(atoms)), 0., len(atoms)
                if relation.kind == "implication":
                    c[i], c[j] = 1., -1.  # A => B: support(A) <= support(B)
                elif relation.kind == "necessary":
                    c[m+j], c[m+i] = 1., -1.  # B necessary for A: refute(B) <= refute(A)
                else:
                    c[i], c[j], d = 1., 1., -1.  # A and B cannot both hold in this context
                c.setflags(write=False)
                scoped = ScopedRule(SoftLogicRule(rid, relation.kind, relation.weight, c, d,
                                                  relation.justification),
                                    (atoms[i].atom_id, atoms[j].atom_id),
                                    *relation.antecedent[1:],
                                    relation.justification + " [catalogue=" + self._rule_catalogue_digest + "]")
                scoped.validate(atoms)
                rules.append(scoped)
            audit.update(status="ACCEPTED_SCOPED_RULES" if rules else "ACCEPTED_NO_APPLICABLE_RULES",
                         available_rule_ids=sorted(available), selected_rule_ids=sorted(selected))
            return ClaimAnalysis(atoms, tuple(gaps), tuple(rules))
        except (ProtocolError, ValueError, KeyError, TypeError) as exc:
            audit.update(status="REJECTED", reason=str(exc))
            if isinstance(exc, ProtocolError):
                raise
            raise ProtocolError(f"VLM 声明/规则解析失败：{exc}") from exc
        finally:
            (self.record_dir/f"analysis-check-{uuid.uuid4().hex}.json").write_text(
                json.dumps(audit, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")

    def score(self, claim, atoms, passages, ledger):
        results = []
        for passage in passages:
            passage.verify()
            data = self._call("score", claim, {"claim": claim.text, "atoms": [asdict(a) for a in atoms],
                                              "passage": self._passage_payload(passage)}, ledger)
            result = ScoredPassage(passage, CanonicalFact(**data["fact"]),
                                   tuple(tuple(float(x) for x in row) for row in data["quality_features"]),
                                   tuple(float(x) for x in data["r_pos"]), tuple(float(x) for x in data["r_neg"]))
            result.validate(len(atoms))
            a = np.asarray(result.quality_features)
            if a.shape != (len(atoms), 3) or np.any(a < 0) or np.any(a > 1):
                raise ProtocolError("质量输入必须是逐原子声明的三项可审计特征")
            results.append(result)
        return tuple(results)

    def terminal(self, state, ledger):
        payload = {"claim": state.claim.text, "atoms": [asdict(a) for a in state.atoms],
                   "evidence": [self._passage_payload(e.passage) for e in state.evidence], "z": state.z, "gaps": state.gaps}
        logits = self._call("terminal", state.claim, payload, ledger, logits=True)
        proposed = LABELS[int(np.argmax(logits))]
        citations = self._call("citations", state.claim, {**payload, "proposed_label": proposed}, ledger)
        return TerminalOutput(tuple(logits), tuple(citations["citation_ids"]), str(citations["rationale"]))

    @staticmethod
    def _passage_payload(passage):
        return {k: v for k,v in asdict(passage).items()
                if k not in {"snapshot_path", "snapshot_sha256", "start", "end"}}

    def with_view(self, view: InputView):
        import copy
        result = copy.copy(self)
        result.view = view
        result.view_version = view.version
        return result
