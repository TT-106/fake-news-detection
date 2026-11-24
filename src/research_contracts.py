"""申请书 2.2 的数据边界。这里不访问数据集答案，也不生成替代证据。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from src.core_models import EvidenceSnippet, SoftLogicRule


class ProtocolError(ValueError):
    """输入不能支持所声明的实验协议。"""


def _fingerprint_value(value: Any, active: set[int]) -> Any:
    """归一化领域对象；普通 JSON 的字节表示保持与既有档案一致。"""
    if isinstance(value, np.generic):
        converted = value.item()
        if isinstance(converted, np.generic):
            raise ProtocolError("无法无损规范化该 NumPy 标量")
        return _fingerprint_value(converted, active)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ProtocolError("指纹输入不得含 NaN 或无穷值")
        return value
    if id(value) in active:
        raise ProtocolError("指纹输入不得含循环引用")
    active.add(id(value))
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return {f.name: _fingerprint_value(getattr(value, f.name), active) for f in fields(value)}
        if isinstance(value, np.ndarray):
            return _fingerprint_value(value.tolist(), active)
        if isinstance(value, dict):
            normalized = {}
            for key, item in value.items():
                key = _fingerprint_value(key, active)
                if key is not None and type(key) not in (str, bool, int, float):
                    raise ProtocolError("指纹对象的键不是 JSON 支持的标量")
                normalized[key] = _fingerprint_value(item, active)
            try:
                sorted(normalized)
            except TypeError as exc:
                raise ProtocolError("混合类型的指纹键不能稳定排序；不进行隐式字符串转换") from exc
            return normalized
        if isinstance(value, (list, tuple)):
            return [_fingerprint_value(item, active) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [_fingerprint_value(item, active) for item in value]
            return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False,
                                                                  allow_nan=False, separators=(",", ":")))
    finally:
        active.remove(id(value))
    raise ProtocolError(f"指纹不支持的对象类型：{type(value).__name__}")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(_fingerprint_value(value, set()), sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_verified_relation(value: Any) -> None:
    if value is not None and type(value) is not bool:
        raise ProtocolError("已核实来源关系只能为 True、False 或 None；概率不得冒充核验结果")


def normalized_url(url: str) -> str:
    p = urlsplit(url.strip())
    if p.scheme.lower() not in ("http", "https") or not p.hostname:
        raise ProtocolError("证据必须提供实际 HTTP(S) URL")
    # 保留有语义的 query；不把同域名、不同时间的网页合并为同一来源。
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path or "/", p.query, ""))


def _unit_vector(values: tuple[float, ...], size: int, name: str) -> None:
    a = np.asarray(values, dtype=float)
    if a.shape != (size,) or not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 1):
        raise ProtocolError(f"{name} 必须为 {size} 维 [0,1] 有限向量")


@dataclass(frozen=True)
class ImageAsset:
    path: str
    sha256: str

    def verify(self) -> None:
        p = Path(self.path)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != self.sha256:
            raise ProtocolError("原图不存在或 SHA-256 不匹配")
        from PIL import Image
        with Image.open(p) as image:
            image.verify()


@dataclass(frozen=True)
class PublicClaim:
    claim_id: str
    event_group: str
    text: str
    images: tuple[ImageAsset, ...]
    date: str = ""
    location: str = ""

    def __post_init__(self):
        if not self.claim_id or not self.event_group or not self.text.strip():
            raise ProtocolError("声明 ID、事件组和配文不能为空")

    @classmethod
    def from_sample(cls, sample: dict, claim_id: str, event_group: str,
                    images: tuple[ImageAsset, ...]) -> PublicClaim:
        # 只从白名单读取信息。QA、article、label、justification 均不进入模型/检索器。
        return cls(claim_id, event_group, sample.get("claim_text") or "", images,
                   sample.get("date") or "", sample.get("location") or "")


@dataclass(frozen=True)
class AtomicClaim:
    atom_id: str
    text: str
    entity: str
    event: str
    time: str = ""
    location: str = ""


@dataclass(frozen=True)
class CanonicalFact:
    entity: str
    event: str
    relation: str
    polarity: str
    time: str
    location: str

    def __post_init__(self):
        if not all(v.strip() for v in (self.entity, self.event, self.relation, self.polarity)):
            raise ProtocolError("规范事实必须包含实体、事件、关系与极性")

    @property
    def key(self) -> str:
        return fingerprint({k: " ".join(v.casefold().split()) for k, v in asdict(self).items()})


@dataclass(frozen=True)
class ArchivedPassage:
    evidence_id: str
    url: str
    snapshot_path: str
    snapshot_sha256: str
    start: int
    end: int
    text: str
    retrieved_at: str
    published_at: str = ""
    author: str = ""
    image_provenance_verified: bool = False
    provenance_proof: str = ""

    def verify(self) -> None:
        normalized_url(self.url)
        p = Path(self.snapshot_path)
        if not self.evidence_id or not p.is_file():
            raise ProtocolError("证据 ID 或归档网页快照缺失")
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.snapshot_sha256:
            raise ProtocolError("网页快照 SHA-256 不匹配")
        content = raw.decode("utf-8")
        if not (0 <= self.start < self.end <= len(content)) or content[self.start:self.end] != self.text:
            raise ProtocolError("引文与归档快照位置不匹配")
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(self.retrieved_at.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise ValueError("timezone required")
        except (ValueError, TypeError) as exc:
            raise ProtocolError("取证时间必须是含时区的 ISO 时间") from exc
        if self.image_provenance_verified and not (self.provenance_proof and self.published_at and self.author):
            raise ProtocolError("图像出处核实必须包含首发时间、作者和证明记录")

    @property
    def exact_identity(self) -> tuple:
        return normalized_url(self.url), self.snapshot_sha256, self.start, self.end


@dataclass(frozen=True)
class ScoredPassage:
    passage: ArchivedPassage
    fact: CanonicalFact
    quality_features: tuple[tuple[float, ...], ...]  # 每个原子声明一行，进入独立质量 LR
    r_pos: tuple[float, ...]
    r_neg: tuple[float, ...]

    def validate(self, m: int):
        self.passage.verify()
        _unit_vector(self.r_pos, m, "r_pos")
        _unit_vector(self.r_neg, m, "r_neg")
        if len(self.quality_features) != m:
            raise ProtocolError("质量模型输入必须逐原子声明提供")
        a = np.asarray(self.quality_features, dtype=float)
        if a.ndim != 2 or not np.all(np.isfinite(a)):
            raise ProtocolError("质量特征必须是有限数值矩阵")


@dataclass(frozen=True)
class ScopedRule:
    rule: SoftLogicRule
    atom_ids: tuple[str, ...]
    entity: str
    event: str
    time: str
    location: str
    justification: str

    def validate(self, atoms: tuple[AtomicClaim, ...]):
        atom_map = {a.atom_id: a for a in atoms}
        if not self.atom_ids or not self.justification.strip():
            raise ProtocolError("软规则必须提供适用原子声明和可靠关系依据")
        for aid in self.atom_ids:
            if aid not in atom_map:
                raise ProtocolError("规则引用了不存在的原子声明")
            a = atom_map[aid]
            if (a.entity, a.event, a.time, a.location) != (self.entity, self.event, self.time, self.location):
                raise ProtocolError("规则的实体/事件/时空条件不一致")
        m = len(atoms)
        c = np.asarray(self.rule.c)
        if c.shape != (2*m,):
            raise ProtocolError("规则状态维度不匹配")
        if self.rule.rule_type not in {"implication", "necessary", "incompatibility"}:
            raise ProtocolError("未知规则类型")
        # 双通道同时较高可以表达可信证据冲突，不施加普遍的正反互斥约束。
        if self.rule.rule_type == "incompatibility" and any(c[j] > 0 and c[m+j] > 0 for j in range(m)):
            raise ProtocolError("不得用通用正反互斥规则消除同一声明的证据冲突")


@dataclass(frozen=True)
class CostEntry:
    call_id: str
    component: str
    amount_rmb: float
    elapsed_ms: float
    success: bool
    request_sha256: str
    billing_basis: str
    input_tokens: int = 0
    output_tokens: int = 0
    replayed: bool = False
    replay_execution_id: str = ""

    def __post_init__(self):
        if (not all((self.call_id, self.component, self.request_sha256, self.billing_basis))
                or not math.isfinite(self.amount_rmb) or self.amount_rmb < 0
                or not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0
                or min(self.input_tokens, self.output_tokens) < 0):
            raise ProtocolError("费用必须有逐请求账单依据，失败请求也要记录")


@dataclass
class CostLedger:
    entries: list[CostEntry] = field(default_factory=list)

    def add(self, entry: CostEntry):
        if entry.call_id in {e.call_id for e in self.entries}:
            raise ProtocolError("同一请求不得重复入账")
        self.entries.append(entry)

    @property
    def total(self) -> float:
        return math.fsum(e.amount_rmb for e in self.entries)


@dataclass(frozen=True)
class CandidateAction:
    action_id: str
    kind: Literal["fact_search", "source_verify", "image_trace"]
    query: str
    expected_cost: float
    cost_upper_bound: float
    cost_basis: str
    target_pair: tuple[int, int] | None = None

    def __post_init__(self):
        if self.kind not in {"fact_search", "source_verify", "image_trace"}:
            raise ProtocolError("未知核验行动")
        if (not self.action_id or self.action_id == "STOP" or not self.query or not self.cost_basis
                or not math.isfinite(self.expected_cost) or not math.isfinite(self.cost_upper_bound)
                or not 0 <= self.expected_cost <= self.cost_upper_bound):
            raise ProtocolError("候选行动须提供预计费用、费用上限和先导核算依据")
        if self.kind == "source_verify" and self.target_pair is None:
            raise ProtocolError("来源核验必须指向当前证据中的具体片段对")


@dataclass(frozen=True)
class HistoryState:
    claim: PublicClaim
    atoms: tuple[AtomicClaim, ...]
    evidence: tuple[ScoredPassage, ...]
    snippets: tuple[EvidenceSnippet, ...]
    source_matrix: tuple[tuple[float, ...], ...]
    z: tuple[float, ...]
    rules: tuple[ScopedRule, ...]
    gaps: tuple[str, ...]
    step: int = 0
    completed_actions: tuple[str, ...] = ()

    @property
    def identity(self) -> str:
        return fingerprint({"claim": asdict(self.claim), "atoms": [asdict(a) for a in self.atoms],
                            "evidence": [{"passage": asdict(e.passage), "fact": e.fact.key,
                                          "quality": s.quality.tolist(), "pos": s.r_pos.tolist(),
                                          "neg": s.r_neg.tolist()}
                                         for e, s in zip(self.evidence, self.snippets)],
                            "P": self.source_matrix, "z": self.z, "gaps": self.gaps,
                            "step": self.step, "completed": self.completed_actions,
                            "rules": [{"id": r.rule.rule_id, "c": r.rule.c.tolist(), "d": r.rule.d,
                                       "weight": r.rule.weight, "scope": r.atom_ids} for r in self.rules]})


@dataclass(frozen=True)
class TerminalOutput:
    logits: tuple[float, float, float, float]
    citation_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class Prediction:
    probabilities: tuple[float, ...]
    raw_label: str
    final_label: str
    sufficiency: float
    citation_ids: tuple[str, ...]
    citations_replayable: bool
    state_id: str

    @property
    def abstained(self) -> bool:
        return self.final_label == "ABSTAIN"
