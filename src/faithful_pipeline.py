"""申请书式(1)—式(4)的执行器。模型和工具通过显式接口注入，无模拟回退。"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import math
import time
import uuid
from typing import Protocol
from urllib.parse import urlsplit

import numpy as np

from src.core_models import (EvidenceSnippet, LABELS, compute_source_influence_feature,
                             solve_convex_qp_inference)
from src.learned_components import (ActionGainRegressor, BranchObservation, ReviewedBinaryModel,
                                    SharedTemperature, chow_loss)
from src.research_contracts import (ArchivedPassage, AtomicClaim, CandidateAction, CostEntry, CostLedger,
                                    HistoryState, Prediction, ProtocolError, PublicClaim,
                                    ScoredPassage, ScopedRule, TerminalOutput, fingerprint,
                                    normalized_url, validate_verified_relation)


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    revision: str
    prompt_sha256: str
    quantization: str

    def __post_init__(self):
        if not all(asdict(self).values()):
            raise ProtocolError("VLM 必须冻结模型版本、提示词与量化设置")


@dataclass(frozen=True)
class ClaimAnalysis:
    atoms: tuple[AtomicClaim, ...]
    gaps: tuple[str, ...]
    rules: tuple[ScopedRule, ...] = ()


class FrozenVisionLanguageModel(Protocol):
    identity: ModelIdentity

    def analyse(self, claim: PublicClaim, ledger: CostLedger) -> ClaimAnalysis: ...
    def score(self, claim: PublicClaim, atoms: tuple[AtomicClaim, ...],
              passages: tuple[ArchivedPassage, ...], ledger: CostLedger) -> tuple[ScoredPassage, ...]: ...
    def terminal(self, state: HistoryState, ledger: CostLedger) -> TerminalOutput: ...


@dataclass(frozen=True)
class RelationObservation:
    target_pair: tuple[int, int]
    same_origin: bool | None
    proof: tuple[ArchivedPassage, ...]

    def __post_init__(self):
        validate_verified_relation(self.same_origin)


@dataclass(frozen=True)
class GapResolution:
    gap: str
    evidence_id: str
    quote: str
    reason: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.gap, self.evidence_id, self.quote, self.reason)):
            raise ProtocolError("解除缺口必须指出具体缺口、新证据 ID、准确引文和理由")


@dataclass(frozen=True)
class ToolOutcome:
    passages: tuple[ArchivedPassage, ...] = ()
    relation: RelationObservation | None = None
    resolved_gaps: tuple[str, ...] = ()
    failed: bool = False
    error: str = ""
    gap_resolutions: tuple[GapResolution, ...] = ()


class EvidenceTools(Protocol):
    """实现须归档真实返回并记录每次成功/失败；接口不接收 dataset QA 或真值。"""
    version: str
    initial_cost_upper_bound: float  # 含初始解析、检索、材料处理、状态更新和初次终端评分

    def initial(self, claim: PublicClaim, analysis: ClaimAnalysis, ledger: CostLedger) -> ToolOutcome: ...
    def candidates(self, state: HistoryState) -> tuple[CandidateAction, ...]: ...
    def execute(self, state: HistoryState, action: CandidateAction, ledger: CostLedger) -> ToolOutcome: ...
    def finalize(self, state: HistoryState, ledger: CostLedger) -> None: ...


def source_pair_features(a: ArchivedPassage, b: ArchivedPassage) -> tuple[float, ...]:
    """可观测引用链、日期、视觉出处和近重复内容；不以域名不同推断独立真值。"""
    ua, ub = normalized_url(a.url), normalized_url(b.url)
    text_a = open_snapshot(a)
    text_b = open_snapshot(b)
    citation_link = float(ua in text_b or ub in text_a or ua == ub)
    same_domain = float(urlsplit(ua).hostname == urlsplit(ub).hostname)
    known_time = bool(a.published_at and b.published_at)
    time_similarity = 0.
    if known_time:
        try:
            ta = datetime.fromisoformat(a.published_at.replace("Z", "+00:00"))
            tb = datetime.fromisoformat(b.published_at.replace("Z", "+00:00"))
            time_similarity = 1. / (1. + abs((ta-tb).total_seconds())/86400.)
        except (ValueError, TypeError):
            known_time = False
    def grams(text):
        text = " ".join(text.casefold().split())
        return {text[i:i+3] for i in range(max(0, len(text)-2))}
    ga, gb = grams(a.text), grams(b.text)
    similarity = len(ga & gb)/max(1, len(ga | gb))
    # 视觉匹配须来自归档的图像出处证明，不使用数据集 claim/input 文件名或 CRC 自重合。
    both_provenance = float(a.image_provenance_verified and b.image_provenance_verified
                            and a.provenance_proof == b.provenance_proof)
    return citation_link, time_similarity, both_provenance, similarity, same_domain, float(known_time)


def open_snapshot(passage: ArchivedPassage) -> str:
    from pathlib import Path
    passage.verify()
    return Path(passage.snapshot_path).read_text(encoding="utf-8")


def feasible_actions(state: HistoryState, candidates: tuple[CandidateAction, ...],
                     spent: float, budget: float, output_reserve: float) -> tuple[CandidateAction, ...]:
    if any(not math.isfinite(v) or v < 0 for v in (spent, budget, output_reserve)):
        raise ProtocolError("预算/累计费用/输出预留必须是非负有限数")
    if state.step >= 3:
        return ()
    ids = [a.action_id for a in candidates]
    if len(ids) != len(set(ids)) or len(candidates) > 3 or len({a.kind for a in candidates}) != len(candidates):
        raise ProtocolError("共享生成器每轮至多生成三类各一个候选行动")
    feasible = []
    for a in candidates:
        if a.target_pair is not None:
            i, j = a.target_pair
            if i == j or not (0 <= i < len(state.evidence) and 0 <= j < len(state.evidence)):
                raise ProtocolError("候选目标必须是当前已经获得的不同证据片段")
        if a.action_id not in state.completed_actions and spent+a.cost_upper_bound+output_reserve <= budget+1e-12:
            feasible.append(a)
    return tuple(feasible)


def choose_one_step(actions: tuple[CandidateAction, ...], predicted_gains: dict[str, float], gamma: float = 1.) -> str:
    if not math.isfinite(gamma) or gamma < 0:
        raise ProtocolError("gamma 必须是非负有限数")
    best, value = "STOP", 0.
    for a in sorted(actions, key=lambda a: a.action_id):
        gain = predicted_gains[a.action_id]
        if not math.isfinite(gain):
            raise ProtocolError("行动收益预测无效")
        net = gain - gamma * a.expected_cost
        # STOP 净收益严格为零；并列时保留 STOP。
        if net > value + 1e-12:
            best, value = a.action_id, net
    return best


def action_features(state: HistoryState, action: CandidateAction, spent: float, budget: float,
                    mode: str = "full", permuted_chi: float | None = None) -> tuple[float, ...]:
    P = np.asarray(state.source_matrix, dtype=float).reshape(len(state.evidence), len(state.evidence))
    if action.target_pair is None:
        chi, iota, uncertainty = 0., 0., 0.
    else:
        i, j = action.target_pair
        p = P[i, j]
        uncertainty = float(p*(1-p))
        iota = 1.
        chi = 0.
        if mode == "full":
            chi, _, _ = compute_source_influence_feature(list(state.snippets), P, action.target_pair,
                                                         rules=[r.rule for r in state.rules])
    off_diag = P[np.triu_indices(len(P), 1)]
    m = len(state.atoms)
    z = np.asarray(state.z)
    base = (len(state.evidence)/8.,
            float(np.mean(off_diag)) if len(off_diag) else 0.,
            float(np.mean(off_diag*(1-off_diag))) if len(off_diag) else 0.,
            len(state.gaps)/max(1, m),
            float(any(e.passage.image_provenance_verified for e in state.evidence)),
            float(np.mean(z[:m])), float(np.mean(z[m:])),
            float(np.mean(np.minimum(z[:m], z[m:]))),
            float(action.kind == "fact_search"), float(action.kind == "source_verify"),
            float(action.kind == "image_trace"), action.expected_cost,
            max(0., budget-spent), state.step/3., float(iota))
    if mode == "no_chi":
        return base
    if mode == "uncertainty":
        return base+(uncertainty,)
    if mode == "permuted_chi":
        if permuted_chi is None or not math.isfinite(permuted_chi) or permuted_chi < 0:
            raise ProtocolError("匹配置换必须提供来自训练组匹配层的冻结 χ，不能用随机常数替代")
        return base+(float(permuted_chi) if iota else 0.,)
    if mode != "full":
        raise ProtocolError("未知行动特征模式")
    return base+(float(chi),)


class FaithfulPipeline:
    def __init__(self, vlm: FrozenVisionLanguageModel, tools: EvidenceTools,
                 source: ReviewedBinaryModel, quality: ReviewedBinaryModel,
                 sufficiency: ReviewedBinaryModel, temperature: SharedTemperature,
                 sufficiency_threshold: float, output_reserve: float, *, calibration_only: bool = False):
        if (source.purpose, quality.purpose, sufficiency.purpose) != ("source", "quality", "sufficiency"):
            raise ProtocolError("来源、质量、充分性模型不可相互替代")
        if not (0 <= sufficiency_threshold <= 1) or not math.isfinite(output_reserve) or output_reserve < 0:
            raise ProtocolError("充分性阈值或输出预留无效")
        if (temperature.temperature is None and not calibration_only) or not tools.version or not math.isfinite(tools.initial_cost_upper_bound) or tools.initial_cost_upper_bound < 0:
            raise ProtocolError("缺少校准模型或真实工具/初始费用上限")
        self.vlm, self.tools = vlm, tools
        self.source, self.quality, self.sufficiency, self.temperature = source, quality, sufficiency, temperature
        self.sufficiency_threshold, self.output_reserve = sufficiency_threshold, output_reserve
        self.calibration_only = calibration_only
        self._locked_version = self.module_version

    @property
    def module_version(self):
        return fingerprint({"vlm": asdict(self.vlm.identity), "tools": self.tools.version,
                            "input_view": getattr(self.vlm, "view_version", "full"),
                            "source": self.source.version, "quality": self.quality.version,
                            "sufficiency": self.sufficiency.version, "temperature": self.temperature.version,
                            "temperature_value": self.temperature.temperature,
                            "suff_threshold": self.sufficiency_threshold, "prob_threshold": 0.75,
                            "output_reserve": self.output_reserve, "mu": 0.05, "alpha": 1.})

    def _check_lock(self):
        if self.module_version != self._locked_version:
            raise ProtocolError("模型/提示词/阈值在冻结后发生变化，须重新训练行动回归并锁定实验")

    def _assemble(self, claim, analysis, evidence, *, step=0, completed=(), relations=None) -> HistoryState:
        m, n = len(analysis.atoms), len(evidence)
        if not 1 <= m <= 4 or len({a.atom_id for a in analysis.atoms}) != m or n > 8:
            raise ProtocolError("申请书限定 1—4 个唯一原子声明、至多 8 个核心证据")
        for r in analysis.rules:
            r.validate(analysis.atoms)
        snippets = []
        for i, e in enumerate(evidence):
            e.validate(m)
            q = self.quality.predict(e.quality_features)
            r_pos, r_neg = np.array(e.r_pos), np.array(e.r_neg)
            q.setflags(write=False)
            r_pos.setflags(write=False)
            r_neg.setflags(write=False)
            snippets.append(EvidenceSnippet(i, urlsplit(e.passage.url).hostname or "", e.passage.url,
                                            e.fact.key, q, r_pos, r_neg,
                                            e.passage.image_provenance_verified,
                                            bool(e.passage.published_at), True))
        P = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                ids = tuple(sorted((evidence[i].passage.evidence_id, evidence[j].passage.evidence_id)))
                if relations and ids in relations:
                    p = relations[ids]
                else:
                    p = float(self.source.predict([source_pair_features(evidence[i].passage, evidence[j].passage)])[0])
                P[i, j] = P[j, i] = p
        if n:
            z, _ = solve_convex_qp_inference(snippets, P, rules=[r.rule for r in analysis.rules])
        else:
            z = np.zeros(2*m)  # 式(2)无证据/无适用规则时的解析最优解；不伪造片段
            if analysis.rules:
                raise ProtocolError("没有证据时不得应用未经当前证据支持的规则")
        return HistoryState(claim, analysis.atoms, tuple(evidence), tuple(snippets),
                            tuple(tuple(float(x) for x in row) for row in P), tuple(z.tolist()),
                            analysis.rules, analysis.gaps, step, completed)

    def _score_new(self, claim, atoms, existing, passages, ledger):
        selected, seen = [], {e.passage.exact_identity for e in existing}
        ids = {e.passage.evidence_id for e in existing}
        for p in passages:
            p.verify()
            if p.exact_identity in seen:
                continue
            if p.evidence_id in ids:
                raise ProtocolError("不同证据不得复用同一 evidence_id")
            seen.add(p.exact_identity)
            ids.add(p.evidence_id)
            if len(existing)+len(selected) < 8:
                selected.append(p)
        if not selected:
            return tuple(existing)
        scored = self.vlm.score(claim, atoms, tuple(selected), ledger)
        if len(scored) != len(selected) or any(s.passage != p for s, p in zip(scored, selected)):
            raise ProtocolError("VLM 不得新增、替换或遗漏实际取得的证据")
        return tuple(existing)+tuple(scored)

    def assemble_logged(self, claim, analysis, evidence, ledger, **kwargs):
        started, success = time.perf_counter(), False
        try:
            result = self._assemble(claim, analysis, evidence, **kwargs)
            success = True
            return result
        finally:
            ledger.add(CostEntry(uuid.uuid4().hex, "local_quality_source_QP_update", 0.,
                                  (time.perf_counter()-started)*1000, success,
                                  fingerprint({"claim": claim.claim_id, "evidence": [e.passage.evidence_id for e in evidence],
                                               "step": kwargs.get("step",0)}),
                                  "本地质量/来源 LR 与 QP 计算，无 API 账单；记录实际本地处理耗时"))

    def initialise(self, claim: PublicClaim, ledger: CostLedger, budget: float) -> HistoryState:
        self._check_lock()
        if not math.isfinite(budget) or budget < 0:
            raise ProtocolError("预算必须是非负有限数")
        if not claim.images:
            raise ProtocolError("多模态正式实验缺少原始图像；不得降级为 NLI/文本实验")
        for image in claim.images:
            image.verify()
        if ledger.total+self.tools.initial_cost_upper_bound+self.output_reserve > budget+1e-12:
            raise ProtocolError("预算不足以覆盖初始检索、处理和预留输出")
        before = ledger.total
        analysis = self.vlm.analyse(claim, ledger)
        outcome = self.tools.initial(claim, analysis, ledger)
        if outcome.relation is not None or outcome.resolved_gaps or outcome.gap_resolutions or (outcome.failed and outcome.passages):
            raise ProtocolError("初始取证返回不一致，不能把失败请求或隐藏关系当作有效证据")
        evidence = self._score_new(claim, analysis.atoms, (), outcome.passages, ledger)
        state = self.assemble_logged(claim, analysis, evidence, ledger)
        if ledger.total-before > self.tools.initial_cost_upper_bound+1e-12:
            raise ProtocolError("初始费用超过核算上限，停止并保留超支账单")
        return state

    def advance(self, state: HistoryState, action: CandidateAction, ledger: CostLedger, budget: float) -> HistoryState:
        self._check_lock()
        if action not in feasible_actions(state, (action,), ledger.total, budget, self.output_reserve):
            raise ProtocolError("行动不满足式(4)预算、轮次或重复执行约束")
        before = ledger.total
        state_before = state.identity
        outcome = self.tools.execute(state, action, ledger)
        if state.identity != state_before:
            raise ProtocolError("工具不得就地修改既有证据、质量、立场或历史")
        if outcome.failed and (outcome.passages or outcome.relation or outcome.resolved_gaps or outcome.gap_resolutions):
            raise ProtocolError("失败请求不能同时声称获得证据或解除了缺口")
        if not set(outcome.resolved_gaps) <= set(state.gaps):
            raise ProtocolError("行动不能用未存在的缺口改写历史")
        if len(set(outcome.resolved_gaps)) != len(outcome.resolved_gaps):
            raise ProtocolError("解除缺口列表不得重复")
        relations = {tuple(sorted((state.evidence[i].passage.evidence_id, state.evidence[j].passage.evidence_id))):
                         state.source_matrix[i][j]
                     for i in range(len(state.evidence)) for j in range(i+1, len(state.evidence))}
        evidence = state.evidence
        if action.kind == "source_verify":
            if outcome.passages or outcome.resolved_gaps or outcome.gap_resolutions:
                raise ProtocolError("来源核验只更新目标关系，不能顺带解锁事实或改写质量/立场")
            rel = outcome.relation
            if rel is not None:
                validate_verified_relation(rel.same_origin)
                if rel.target_pair != action.target_pair or not rel.proof:
                    raise ProtocolError("来源核验缺少目标一致的实际证明")
                for proof in rel.proof:
                    proof.verify()
                if rel.same_origin is not None:
                    i, j = rel.target_pair
                    key = tuple(sorted((state.evidence[i].passage.evidence_id, state.evidence[j].passage.evidence_id)))
                    relations[key] = float(rel.same_origin)
        else:
            if outcome.relation is not None:
                raise ProtocolError("检索行动不能伪装为已核实来源标注")
            evidence = self._score_new(state.claim, state.atoms, evidence, outcome.passages, ledger)
        old_ids = {e.passage.evidence_id for e in state.evidence}
        accepted_new = {e.passage.evidence_id: e.passage for e in evidence if e.passage.evidence_id not in old_ids}
        proofs = outcome.gap_resolutions
        if any(not isinstance(proof, GapResolution) for proof in proofs):
            raise ProtocolError("缺口证明必须为 GapResolution")
        if {proof.gap for proof in proofs} != set(outcome.resolved_gaps):
            raise ProtocolError("每个解除的缺口必须有明确绑定的新证据证明，不能仅声明已解除")
        for proof in proofs:
            passage = accepted_new.get(proof.evidence_id)
            if passage is None or proof.quote not in passage.text:
                raise ProtocolError("缺口证明必须引用本行动实际接纳的新证据及其准确文本")
        analysis = ClaimAnalysis(state.atoms, tuple(g for g in state.gaps if g not in outcome.resolved_gaps), state.rules)
        result = self.assemble_logged(state.claim, analysis, evidence, ledger, step=state.step+1,
                                completed=state.completed_actions+(action.action_id,), relations=relations)
        if ledger.total-before > action.cost_upper_bound+1e-12:
            raise ProtocolError("行动实际费用超过上限，停止并保留超支账单")
        return result

    def predict(self, state: HistoryState, ledger: CostLedger) -> Prediction:
        self._check_lock()
        if self.calibration_only:
            raise ProtocolError("独立校准过程只能采集原始 VLM logits，不能生成已校准预测")
        state_before = state.identity
        output = self.vlm.terminal(state, ledger)
        if state.identity != state_before:
            raise ProtocolError("终端模型不得修改推理历史")
        probs = self.temperature.probabilities(output.logits)
        label = LABELS[int(np.argmax(probs))]
        ids = tuple(output.citation_ids)
        evidence = {e.passage.evidence_id: e.passage for e in state.evidence}
        replay_ok = bool(ids) and len(set(ids)) == len(ids) and all(cid in evidence for cid in ids)
        if replay_ok:
            for cid in ids:
                evidence[cid].verify()
        m = len(state.atoms)
        z = np.asarray(state.z)
        cited = [e for e in state.evidence if e.passage.evidence_id in ids]
        features = (min(1., len({e.fact.key for e in cited})/m),
                    float(any(e.passage.image_provenance_verified for e in cited)),
                    float(np.mean(z[:m])), float(np.mean(z[m:])),
                    float(np.mean(np.minimum(z[:m], z[m:]))), max(probs), float(replay_ok))
        sufficiency = float(self.sufficiency.predict([features])[0])
        accept = max(probs) > 0.75 and sufficiency >= self.sufficiency_threshold and replay_ok
        return Prediction(probs, label, label if accept else "ABSTAIN", sufficiency, ids, replay_ok, state.identity)

    def run_episode(self, claim: PublicClaim, regressor: ActionGainRegressor,
                    budget: float = 0.25, gamma: float = 1.) -> dict:
        episode_start = time.perf_counter()
        if not math.isfinite(gamma) or gamma < 0:
            raise ProtocolError("成本交换系数必须是非负有限数")
        ledger = CostLedger()
        state = self.initialise(claim, ledger, budget)
        prediction = self.predict(state, ledger)
        if ledger.total > self.tools.initial_cost_upper_bound+1e-12:
            raise ProtocolError("含首次终端判定的完整初始处理超过费用上限")
        trajectory = []
        for _ in range(3):
            actions = feasible_actions(state, self.tools.candidates(state), ledger.total, budget, self.output_reserve)
            if hasattr(regressor, "choose"):
                gains = {}
                chosen = regressor.choose(state, prediction, actions)
                if chosen not in {"STOP"} | {a.action_id for a in actions}:
                    raise ProtocolError("基线路由选择了预算不可行或不存在的行动")
            else:
                gains = {a.action_id: regressor.predict_gain(
                    regressor.features_for(state, a, ledger.total, budget) if hasattr(regressor, "features_for")
                    else action_features(state, a, ledger.total, budget, regressor.feature_mode)) for a in actions}
                chosen = choose_one_step(actions, gains, gamma)
            trajectory.append({"history_id": state.identity, "action": chosen, "predicted_gains": gains,
                               "spent_before": ledger.total})
            if chosen == "STOP":
                break
            a = next(a for a in actions if a.action_id == chosen)
            before = ledger.total
            state = self.advance(state, a, ledger, budget)
            prediction = self.predict(state, ledger)
            if ledger.total-before > a.cost_upper_bound+1e-12:
                raise ProtocolError("完整行动(含材料处理/终端更新)超出费用上限")
        before_output = ledger.total
        self.tools.finalize(state, ledger)
        if ledger.total-before_output > self.output_reserve+1e-12 or ledger.total > budget+1e-12:
            raise ProtocolError("输出或总费用超出预算，不能将超支运行计作预算内结果")
        return {"claim_id": claim.claim_id, "event_group": claim.event_group,
                "prediction": asdict(prediction), "trajectory": trajectory,
                "ledger": [asdict(e) for e in ledger.entries], "total_cost_rmb": ledger.total,
                "episode_elapsed_ms": (time.perf_counter()-episode_start)*1000,
                "module_version": self._locked_version, "state": state}

    def common_history_branches(self, state: HistoryState, ledger: CostLedger, true_label: str,
                                budget: float = 0.25, mode: str = "full") -> list[BranchObservation]:
        """所有候选从相同不可变历史展开。真值只参与离线 Chow 计算，不进入执行器。"""
        before_prediction = self.predict(state, ledger)
        before_loss = chow_loss(before_prediction, true_label)
        spent = ledger.total
        actions = feasible_actions(state, self.tools.candidates(state), spent, budget, self.output_reserve)
        rows = []
        for a in actions:
            features = action_features(state, a, spent, budget, mode)
            branch_ledger = copy.deepcopy(ledger)
            next_state = self.advance(state, a, branch_ledger, budget)
            after_prediction = self.predict(next_state, branch_ledger)
            realized_cost = branch_ledger.total-spent
            if realized_cost > a.cost_upper_bound+1e-12:
                raise ProtocolError("共同历史分支实际总费用超过候选上限")
            rows.append(BranchObservation(state.claim.claim_id, state.claim.event_group, state.identity,
                                          a.action_id, features, before_loss, chow_loss(after_prediction, true_label),
                                          a.expected_cost, realized_cost))
        rows.append(BranchObservation(state.claim.claim_id, state.claim.event_group, state.identity,
                                      "STOP", (), before_loss, before_loss, 0., 0.))
        return rows
