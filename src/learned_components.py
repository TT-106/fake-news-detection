"""独立来源/质量/充分性 LR、共享温度及按事件分组的行动收益岭回归。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace as dataclass_replace
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core_models import LABELS
from src.research_contracts import Prediction, ProtocolError, fingerprint


@dataclass(frozen=True)
class ReviewedLabel:
    reviewer_a: str
    reviewer_b: str
    label_a: int | None
    label_b: int | None
    rationale: str
    adjudicator: str = ""
    adjudicated_label: int | None = None
    reviewer_type: str = "human"

    def resolve(self) -> int | None:
        if (not self.reviewer_a or not self.reviewer_b or self.reviewer_a == self.reviewer_b
                or self.reviewer_type != "human" or not self.rationale.strip()):
            raise ProtocolError("来源、质量和充分性监督须有双人独立人工标注依据")
        for v in (self.label_a, self.label_b, self.adjudicated_label):
            if v not in (None, 0, 1):
                raise ProtocolError("关系标注只允许 1=同源/0=独立/None=无法核实")
        if self.label_a == self.label_b:
            if self.adjudicator or self.adjudicated_label is not None:
                raise ProtocolError("一致标注不应混入第三个覆盖标签")
            return self.label_a
        if not self.adjudicator or self.adjudicator in {self.reviewer_a, self.reviewer_b}:
            raise ProtocolError("不一致标注必须由第三人仲裁")
        return self.adjudicated_label


@dataclass(frozen=True)
class SupervisedRow:
    claim_id: str
    event_group: str
    features: tuple[float, ...]
    label: ReviewedLabel


@dataclass(frozen=True)
class ExperimentSplit:
    module_fit: frozenset[str]
    temperature_calibration: frozenset[str]
    pilot: frozenset[str]
    action_train: frozenset[str]
    mechanism_validation: frozenset[str]
    final_eval: frozenset[str]
    reserve: frozenset[str] = frozenset()

    def validate(self, event_groups: dict[str, str]) -> None:
        seen_ids, seen_groups = set(), {}
        for role, ids in asdict(self).items():
            if seen_ids & ids:
                raise ProtocolError("同一声明被分配到多个实验用途")
            seen_ids |= ids
            for cid in ids:
                if cid not in event_groups or not event_groups[cid]:
                    raise ProtocolError("缺少可审核的事件分组")
                group = event_groups[cid]
                if group in seen_groups and seen_groups[group] != role:
                    raise ProtocolError("事件组跨拟合/校准/训练/验证/评测集合泄漏")
                seen_groups[group] = role


def grouped_folds(groups: np.ndarray, y: np.ndarray, classification: bool) -> list:
    """同一声明的片段、证据对和分支始终留在同一折。缺样本时明确拒绝拟合。"""
    classes = set(y.tolist())
    for n in range(min(5, len(set(groups.tolist()))), 1, -1):
        folds = list(GroupKFold(n_splits=n, shuffle=True, random_state=42).split(np.zeros((len(y), 1)), y, groups))
        if not classification or all(set(y[tr].tolist()) == classes and set(y[va].tolist()) == classes for tr, va in folds):
            return folds
    raise ProtocolError("独立事件组/已核实正负例不足，不能进行无泄漏分组交叉验证")


class ReviewedBinaryModel:
    """同一实现的三个独立实例，参数、训练记录和用途不能互相代替。"""

    def __init__(self, purpose: str):
        if purpose not in {"source", "quality", "sufficiency"}:
            raise ProtocolError("未知轻量模型用途")
        self.purpose = purpose
        self.estimator = None
        self.training_claims: frozenset[str] = frozenset()
        self.training_groups: frozenset[str] = frozenset()
        self.training_digest = ""

    def fit(self, rows: list[SupervisedRow], split: ExperimentSplit, event_groups: dict[str, str]):
        split.validate(event_groups)
        if not rows or any(r.claim_id not in split.module_fit or event_groups.get(r.claim_id) != r.event_group for r in rows):
            raise ProtocolError("来源/质量/充分性模型只能使用模块开发拟合集中的标注")
        verified = [(r, r.label.resolve()) for r in rows]
        verified = [(r, y) for r, y in verified if y is not None]
        if not verified or {y for _, y in verified} != {0, 1}:
            raise ProtocolError("模型缺少经核实的正例和负例；无法核实关系不得当作独立关系")
        X = np.asarray([r.features for r, _ in verified], dtype=float)
        y = np.asarray([y for _, y in verified], dtype=int)
        if X.ndim != 2 or not np.all(np.isfinite(X)):
            raise ProtocolError("模型特征必须是有限数值矩阵")
        groups = np.asarray([r.event_group for r, _ in verified])
        folds = grouped_folds(groups, y, classification=True)
        pipe = Pipeline([("scale", StandardScaler()),
                         ("model", LogisticRegression(max_iter=1000, random_state=42))])
        search = GridSearchCV(pipe, {"model__C": [0.1, 1.0, 10.0]}, cv=folds,
                              scoring="neg_log_loss", error_score="raise")
        search.fit(X, y)
        self.estimator = search.best_estimator_
        self.training_claims = frozenset(r.claim_id for r, _ in verified)
        self.training_groups = frozenset(groups.tolist())
        self.training_digest = fingerprint([asdict(r) for r, _ in verified])
        return self

    def predict(self, X) -> np.ndarray:
        if self.estimator is None:
            raise ProtocolError(f"{self.purpose} 模型尚未用真实标注拟合")
        a = np.asarray(X, dtype=float)
        if a.ndim != 2 or not np.all(np.isfinite(a)):
            raise ProtocolError("预测特征无效")
        return self.estimator.predict_proba(a)[:, 1]

    @property
    def version(self) -> str:
        if self.estimator is None:
            raise ProtocolError("未拟合模型不能冻结")
        lr, scale = self.estimator["model"], self.estimator["scale"]
        return fingerprint({"purpose": self.purpose, "data": self.training_digest,
                            "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
                            "classes": lr.classes_.tolist(),
                            "scale": scale.scale_.tolist(), "mean": scale.mean_.tolist()})


@dataclass(frozen=True)
class CalibrationRow:
    claim_id: str
    event_group: str
    logits: tuple[float, float, float, float]
    true_label: str


class SharedTemperature:
    def __init__(self):
        self.temperature = None
        self.version = ""

    def fit(self, rows: list[CalibrationRow], split: ExperimentSplit, event_groups: dict[str, str]):
        split.validate(event_groups)
        if not rows or any(r.claim_id not in split.temperature_calibration
                           or event_groups.get(r.claim_id) != r.event_group for r in rows):
            raise ProtocolError("温度只能在独立的开发校准事件组上拟合")
        logits = np.asarray([r.logits for r in rows], dtype=float)
        if logits.shape != (len(rows), 4) or not np.all(np.isfinite(logits)):
            raise ProtocolError("校准必须提供按固定四类顺序排列的真实 VLM 原始得分")
        y = np.asarray([LABELS.index(r.true_label) for r in rows])
        if len(set(y.tolist())) < 2:
            raise ProtocolError("温度校准不能只含单一类别")
        # 同一声明的多个历史不增加该声明在校准中的权重。
        counts = {cid: sum(r.claim_id == cid for r in rows) for cid in {r.claim_id for r in rows}}
        weights = np.asarray([1.0/counts[r.claim_id] for r in rows])

        def nll(log_t):
            scaled = logits / np.exp(log_t)
            return np.average(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y], weights=weights)

        result = minimize_scalar(nll, bounds=(-4., 4.), method="bounded")
        if not result.success:
            raise ProtocolError("温度拟合失败")
        self.temperature = float(np.exp(result.x))
        self.version = fingerprint({"rows": [asdict(r) for r in rows], "T": self.temperature})
        return self

    def probabilities(self, logits) -> tuple[float, ...]:
        if self.temperature is None:
            raise ProtocolError("共享温度尚未独立校准")
        a = np.asarray(logits, dtype=float)
        if a.shape != (4,) or not np.all(np.isfinite(a)):
            raise ProtocolError("终端 VLM 必须输出四个有限原始得分")
        return tuple(float(x) for x in softmax(a / self.temperature))


def chow_loss(prediction: Prediction, true_label: str) -> float:
    """唯一的行动收益监督损失；真值只在预测完成后的离线评价中使用。"""
    if true_label not in LABELS:
        raise ProtocolError("未知评价标签")
    return 0.25 if prediction.abstained else float(prediction.final_label != true_label)


@dataclass(frozen=True)
class BranchObservation:
    claim_id: str
    event_group: str
    history_id: str
    action_id: str
    features: tuple[float, ...]
    loss_before: float
    loss_after: float
    expected_cost: float
    realized_cost: float
    target_uncertainty: float = 0.

    @property
    def gain(self) -> float:
        if not math.isfinite(self.target_uncertainty) or not 0 <= self.target_uncertainty <= .25+1e-12:
            raise ProtocolError("目标关系不确定度超出定义域")
        if self.loss_before not in {0., 0.25, 1.} or self.loss_after not in {0., 0.25, 1.}:
            raise ProtocolError("收益监督必须是式(4)约定的 Chow 损失差，不能混入连续风险或 χ 奖励")
        if self.action_id == "STOP" and self.loss_before != self.loss_after:
            raise ProtocolError("STOP 收益必须严格为零")
        return self.loss_before - self.loss_after


class ActionGainRegressor:
    def __init__(self, feature_mode: str = "full"):
        if feature_mode not in {"full", "no_chi", "uncertainty", "permuted_chi"}:
            raise ProtocolError("未知行动特征消融")
        self.feature_mode = feature_mode
        self.estimator = None
        self.version = ""
        self.permutation_control = None

    def fit(self, rows: list[BranchObservation], split: ExperimentSplit, event_groups: dict[str, str], frozen_modules: str):
        split.validate(event_groups)
        if not frozen_modules:
            raise ProtocolError("必须先锁定来源、质量、VLM、规则和终端模块")
        if not rows or any(r.claim_id not in split.action_train or event_groups.get(r.claim_id) != r.event_group for r in rows):
            raise ProtocolError("行动收益回归只能使用 action_train 中的共同历史分支")
        # STOP 不通过回归估计，部署时固定为零；保留其原始记录以便审计。
        usable = [r for r in rows if r.action_id != "STOP"]
        for row in rows:
            _ = row.gain
            if not all(math.isfinite(v) and v >= 0 for v in (row.expected_cost, row.realized_cost)):
                raise ProtocolError("分支费用无效")
        if not usable:
            raise ProtocolError("没有可用于拟合的真实行动分支")
        X = np.asarray([r.features for r in usable], dtype=float)
        y = np.asarray([r.gain for r in usable], dtype=float)
        if X.ndim != 2 or not np.all(np.isfinite(X)):
            raise ProtocolError("行动特征无效")
        groups = np.asarray([r.event_group for r in usable])
        folds = grouped_folds(groups, y, classification=False)
        pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
        search = GridSearchCV(pipe, {"model__alpha": [0.1, 1., 10., 100.]},
                              scoring="neg_mean_squared_error", cv=folds, error_score="raise")
        search.fit(X, y)
        self.estimator = search.best_estimator_
        self.version = fingerprint({"rows": [asdict(r) for r in rows], "modules": frozen_modules,
                                    "mode": self.feature_mode, "alpha": search.best_params_,
                                    "coef": self.estimator["model"].coef_.tolist()})
        return self

    def predict_gain(self, features: tuple[float, ...]) -> float:
        if self.estimator is None:
            raise ProtocolError("行动收益回归器尚未用真实轨迹训练")
        a = np.asarray(features, dtype=float).reshape(1, -1)
        if not np.all(np.isfinite(a)):
            raise ProtocolError("行动特征不能含 NaN/Inf")
        return float(self.estimator.predict(a)[0])

    def features_for(self, state, action, spent, budget):
        from src.faithful_pipeline import action_features
        override = None
        if self.feature_mode == "permuted_chi":
            if self.permutation_control is None:
                raise ProtocolError("匹配置换模型缺少冻结训练供体池")
            if action.target_pair is not None:
                i,j = action.target_pair
                p = state.source_matrix[i][j]
                override = self.permutation_control.donor(state.identity, action.action_id, p*(1-p))
            else:
                override = 0.
        return action_features(state, action, spent, budget, self.feature_mode, override)


class MatchedChiPermutation:
    """冻结训练组的不确定性分层供体池；验证/在线阶段不读取其他评测声明。"""
    def __init__(self, seed=42):
        self.seed = seed
        self.pools = {}
        self.version = ""

    @staticmethod
    def stratum(uncertainty):
        if not math.isfinite(uncertainty) or not 0 <= uncertainty <= .25+1e-12:
            raise ProtocolError("来源关系不确定度超出 p(1-p) 定义域")
        return min(4, int(uncertainty/.05))

    def fit_transform(self, rows: list[BranchObservation]):
        buckets = {}
        for i,r in enumerate(rows):
            if r.action_id != "STOP" and r.features[-2] == 1.:
                buckets.setdefault(self.stratum(r.target_uncertainty), []).append(i)
        if not buckets:
            raise ProtocolError("没有包含目标来源对的训练行动，匹配置换对照不可用")
        rng = np.random.default_rng(self.seed)
        transformed = list(rows)
        for key, indices in sorted(buckets.items()):
            values = [rows[i].features[-1] for i in indices]
            self.pools[key] = tuple(values)
            for index, donor in zip(indices, rng.permutation(values)):
                r = rows[index]
                transformed[index] = dataclass_replace(r, features=r.features[:-1]+(float(donor),))
        self.version = fingerprint({"seed": self.seed, "pools": self.pools,
                                    "training_claims": sorted({r.claim_id for r in rows})})
        return transformed

    def donor(self, history_id, action_id, uncertainty):
        if not self.pools:
            raise ProtocolError("匹配置换供体池未拟合")
        target = self.stratum(uncertainty)
        # 空层采用最近非空训练层，不使用验证/最终评测分布补全。
        key = min(self.pools, key=lambda k: (abs(k-target), k))
        pool = self.pools[key]
        index = int(fingerprint({"seed": self.seed, "history": history_id, "action": action_id})[:16], 16)%len(pool)
        return float(pool[index])
