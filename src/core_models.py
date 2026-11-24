r"""
核心数学模型实现 (Core Mathematical Models for Multimodal Fact-Checking)
对应申请书 2.2 节式(1)—式(4)：
  - 式(1): 来源依赖感知的同源概率证据权重折扣模型 w_{jl}(P)
  - 式(2): 支持/反驳双通道软逻辑凸二次规划一致性推断模型 z*(E, P)
  - 式(3): 面向候选目标的来源影响特征 (反事实状态扰动) chi_t(a)
  - 式(4): 证据充分性约束下的Chow拒识损失与成本敏感一步前瞻决策模型
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.optimize import minimize


@dataclass
class EvidenceSnippet:
    """结构化事实证据片段"""
    snippet_id: int
    domain: str
    url: str
    fact_id: int | str  # 规范事实编号，包含实体/事件/关系/极性/时空限定，不能用槽位代替
    quality: np.ndarray  # 对各原子声明的质量评分 q_{j\ell} \in [0, 1], shape=(m,)
    r_pos: np.ndarray    # 对各原子声明的直接支持评分 r_{j\ell}^+ \in [0, 1], shape=(m,)
    r_neg: np.ndarray    # 对各原子声明的直接反驳评分 r_{j\ell}^- \in [0, 1], shape=(m,)
    is_image_provenance: bool = False
    timestamp_verified: bool = False
    url_replayable: bool = False


@dataclass
class SoftLogicRule:
    r"""
    跨模态软逻辑规则约束:
    线性违反量 \ell_r(z) = c_r^T z + d_r
    惩罚项: \omega_r * max(0, c_r^T z + d_r)^2
    其中状态向量 z = [z_1^+, ..., z_m^+, z_1^-, ..., z_m^-]^T \in [0, 1]^{2m}
    """
    rule_id: str
    rule_type: str  # 'implication' (蕴涵), 'necessary' (必要条件), 'incompatibility' (不相容)
    weight: float   # \omega_r >= 0
    c: np.ndarray   # shape=(2m,)
    d: float        # 标量偏置
    description: str = ""

    def violation(self, z: np.ndarray) -> float:
        return float(max(0.0, np.dot(self.c, z) + self.d))


def validate_evidence(snippets: List[EvidenceSnippet], P: np.ndarray) -> None:
    """显式检查式(1)—式(3)的定义域，禁止静默裁剪坏输入。"""
    n = len(snippets)
    p = np.asarray(P, dtype=float)
    if p.shape != (n, n) or not np.all(np.isfinite(p)):
        raise ValueError("P must be a finite n x n matrix")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("Source probabilities must be in [0, 1]")
    if not np.allclose(p, p.T, atol=1e-12, rtol=0):
        raise ValueError("Source probability matrix must be symmetric")
    if not np.allclose(np.diag(p), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Source probability matrix must have a unit diagonal")
    if not snippets:
        return
    m = len(snippets[0].quality)
    if not 1 <= m <= 4:
        raise ValueError("Expected 1 to 4 atomic claims")
    for s in snippets:
        for name in ("quality", "r_pos", "r_neg"):
            v = np.asarray(getattr(s, name), dtype=float)
            if v.shape != (m,) or not np.all(np.isfinite(v)) or np.any(v < 0) or np.any(v > 1):
                raise ValueError(f"{name} must be a finite vector of length {m} in [0, 1]")


def _positive_mu(mu: float) -> None:
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("mu must be finite and strictly positive for strong convexity")


def compute_discounted_weights(
    snippets: List[EvidenceSnippet],
    P: np.ndarray,
    mode: str = "prob_discount",
    hard_threshold: float = 0.5,
) -> np.ndarray:
    r"""
    计算证据权重矩阵 W \in R^{m \times n}，对应式(1)及表2中的E1对比基线：
      w_{j\ell}(P) = q_{j\ell} / (1 + \sum_{h \in B_\ell, h \neq \ell} p_{\ell h})

    参数:
      snippets: 长度为 n 的证据片段列表，每个片段含 shape=(m,) 的质量向量 q_{\cdot \ell}
      P: shape=(n, n) 的对称同源概率矩阵，对角线为 1，分量 \in [0, 1]
      mode:
        - 'prob_discount': 式(1) 同源概率折扣（本项目方法，或者是 oracle_source 当传入真实P时）
        - 'independent': 传统独立性假设，不作任何同源折扣 w_{j\ell} = q_{j\ell}
        - 'domain_dedup': 粗粒度按网站域名去重（同一域名仅保留整体质量最高的一条片段，其余置0，会误删同域新增事实）
        - 'semantic_dedup': 语义/事实硬阈值去重（在相同事实簇 B_\ell 内若 p_{\ell h} >= hard_threshold 则做硬计数折扣）
        - 'max_quality_single': 相同规范事实簇 B_\ell 内仅保留单条最高质量证据，其余置0
    返回:
      W: shape=(m, n) 的有效证据权重矩阵
    """
    validate_evidence(snippets, P)
    if not np.isfinite(hard_threshold) or not 0 <= hard_threshold <= 1:
        raise ValueError("hard_threshold must be in [0, 1]")
    n = len(snippets)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    m = len(snippets[0].quality)
    Q = np.column_stack([s.quality for s in snippets])  # shape=(m, n)
    fact_ids = [s.fact_id for s in snippets]
    domains = [s.domain for s in snippets]

    W = np.zeros_like(Q, dtype=float)

    if mode == "independent":
        return Q.copy()

    elif mode == "domain_dedup":
        # 按域名粗粒度合并：每个域名仅保留平均质量最高的一条记录
        unique_domains = sorted(set(domains))
        for dom in unique_domains:
            idxs = [i for i, d in enumerate(domains) if d == dom]
            best_idx = max(idxs, key=lambda idx: float(np.mean(Q[:, idx])))
            W[:, best_idx] = Q[:, best_idx]
        return W

    elif mode == "max_quality_single":
        # 每个规范事实编号 B_\ell 内对各声明 j 仅保留质量最高的一条记录
        unique_facts = list(dict.fromkeys(fact_ids))
        for fid in unique_facts:
            idxs = [i for i, f in enumerate(fact_ids) if f == fid]
            for j in range(m):
                best_idx = max(idxs, key=lambda idx: float(Q[j, idx]))
                W[j, best_idx] = Q[j, best_idx]
        return W

    elif mode == "semantic_dedup":
        # 硬阈值计数折扣：在相同规范事实簇内，p_{\ell h} >= hard_threshold 视为完全同源(1)，否则为0
        for ell in range(n):
            same_fact_neighbors = [
                h for h in range(n)
                if h != ell and fact_ids[h] == fact_ids[ell] and P[ell, h] >= hard_threshold
            ]
            denom = 1.0 + float(len(same_fact_neighbors))
            W[:, ell] = Q[:, ell] / denom
        return W

    elif mode == "prob_discount":
        # 式(1): w_{j\ell}(P) = q_{j\ell} / (1 + \sum_{h \in B_\ell, h \neq \ell} p_{\ell h})
        for ell in range(n):
            same_fact_sum = sum(
                float(P[ell, h])
                for h in range(n)
                if h != ell and fact_ids[h] == fact_ids[ell]
            )
            denom = 1.0 + same_fact_sum
            W[:, ell] = Q[:, ell] / denom
        return W

    else:
        raise ValueError(f"Unknown weight discounting mode: {mode}")


def solve_convex_qp_inference(
    snippets: List[EvidenceSnippet],
    P: np.ndarray,
    rules: Optional[List[SoftLogicRule]] = None,
    mu: float = 0.05,
    alpha: float = 1.0,
    weight_mode: str = "prob_discount",
    hard_threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    r"""
    求解式(2) 支持与反驳双通道凸二次规划推断模型:
      A(z; E, P) = 1/2 \sum_{j,\ell} w_{j\ell}(P) ( [r_{j\ell}^+ - z_j^+]_+^2 + [r_{j\ell}^- - z_j^-]_+^2 )
      F(z; E, P) = (\mu / 2) ||z||_2^2 + A(z; E, P) + (\alpha / 2) \sum_{r \in R} \omega_r [ \ell_r(z) ]_+^2
      z*(E, P) = \argmin_{z \in [0, 1]^{2m}} F(z; E, P)

    返回:
      z_star: shape=(2m,) 的全局最优状态向量 [z_1^+, ..., z_m^+, z_1^-, ..., z_m^-]
      info: 包含目标函数值、直接证据拟合项、规则惩罚项、权重矩阵等诊断指标
    """
    _positive_mu(mu)
    validate_evidence(snippets, P)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if rules is None:
        rules = []
    n = len(snippets)
    if n == 0:
        raise ValueError("snippets list cannot be empty")
    m = len(snippets[0].quality)
    for rule in rules:
        if not np.isfinite(rule.weight) or rule.weight < 0:
            raise ValueError("Soft logic rule weights must be finite and nonnegative")
        if np.asarray(rule.c).shape != (2 * m,) or not np.all(np.isfinite(rule.c)) or not np.isfinite(rule.d):
            raise ValueError("Invalid linear rule coefficients")

    W = compute_discounted_weights(
        snippets, P, mode=weight_mode, hard_threshold=hard_threshold
    )  # shape=(m, n)
    R_pos = np.column_stack([s.r_pos for s in snippets])  # shape=(m, n)
    R_neg = np.column_stack([s.r_neg for s in snippets])  # shape=(m, n)

    def objective_and_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        z_pos = z[:m]
        z_neg = z[m:]

        # 正则项 (\mu / 2) ||z||_2^2
        reg_val = 0.5 * mu * float(np.dot(z, z))
        grad = mu * z.copy()

        # 直接证据拟合项 A(z; E, P)
        # diff_pos[j, ell] = max(0, r_{j,ell}^+ - z_j^+)
        diff_pos = np.maximum(0.0, R_pos - z_pos[:, None])
        diff_neg = np.maximum(0.0, R_neg - z_neg[:, None])

        fit_val = 0.5 * float(np.sum(W * (diff_pos ** 2 + diff_neg ** 2)))
        grad_pos = -np.sum(W * diff_pos, axis=1)
        grad_neg = -np.sum(W * diff_neg, axis=1)
        grad[:m] += grad_pos
        grad[m:] += grad_neg

        # 软逻辑规则违反惩罚项 (\alpha / 2) \sum_r \omega_r [\ell_r(z)]_+^2
        rule_val = 0.0
        if alpha > 0 and rules:
            for r in rules:
                viol = float(np.dot(r.c, z) + r.d)
                if viol > 0.0:
                    rule_val += 0.5 * alpha * r.weight * (viol ** 2)
                    grad += alpha * r.weight * viol * r.c

        total_val = reg_val + fit_val + rule_val
        return total_val, grad

    z0 = np.full(2 * m, 0.25, dtype=float)
    bounds = [(0.0, 1.0)] * (2 * m)
    res = minimize(
        objective_and_grad,
        z0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 500},
    )
    if not res.success or not np.all(np.isfinite(res.x)) or not np.isfinite(res.fun):
        raise RuntimeError(f"QP optimization failed: {res.message}")
    z_star = np.clip(res.x, 0.0, 1.0)
    total_val, gradient = objective_and_grad(z_star)
    residual = float(np.max(np.abs(z_star - np.clip(z_star - gradient, 0.0, 1.0))))
    if residual > 1e-5:
        raise RuntimeError(f"QP projected-gradient residual too large: {residual}")

    # 拆解各项用于分析
    z_pos, z_neg = z_star[:m], z_star[m:]
    diff_pos = np.maximum(0.0, R_pos - z_pos[:, None])
    diff_neg = np.maximum(0.0, R_neg - z_neg[:, None])
    fit_val = 0.5 * float(np.sum(W * (diff_pos ** 2 + diff_neg ** 2)))
    rule_val = 0.0
    if alpha > 0 and rules:
        for r in rules:
            v = r.violation(z_star)
            rule_val += 0.5 * alpha * r.weight * (v ** 2)

    info = {
        "F_total": total_val,
        "A_fit": fit_val,
        "R_rule": rule_val,
        "reg": 0.5 * mu * float(np.dot(z_star, z_star)),
        "effective_weight_sum": float(np.sum(W)),
        "projected_gradient_residual": residual,
        "solver_success": True,
    }
    return z_star, info


def theoretical_stability_bound(
    snippets: List[EvidenceSnippet],
    P1: np.ndarray,
    P2: np.ndarray,
    mu: float = 0.05,
) -> float:
    r"""
    计算来源概率估计误差对凸二次规划最优状态的理论稳定性上界:
    由于目标函数 F(z; E, P) 在凸集 [0, 1]^{2m} 上是 \mu-强凸函数，
    且对任意 z \in [0, 1]^{2m}、r \in [0, 1]，|[r - z]_+| <= 1，
    由强凸变分不等式可得状态扰动上界:
      ||z*(E, P1) - z*(E, P2)||_2 <= (1 / \mu) * || \nabla_z A(z*(E,P1); E, P1) - \nabla_z A(z*(E,P1); E, P2) ||_2
                                   <= (\sqrt{2} / \mu) * || (W(P1) - W(P2)) ||_{2,1}
    返回该理论稳定性上界，用于与实际状态漂移 ||z*(E, P1) - z*(E, P2)||_2 进行核验对比。
    """
    _positive_mu(mu)
    W1 = compute_discounted_weights(snippets, P1, mode="prob_discount")
    W2 = compute_discounted_weights(snippets, P2, mode="prob_discount")
    delta_W = np.abs(W1 - W2)  # shape=(m, n)
    row_sums = np.sum(delta_W, axis=1)
    # 每个声明的支持与反驳两通道梯度差绝对值之和均不超过 row_sums[j]
    grad_diff_bound = float(np.sqrt(2.0 * np.sum(row_sums ** 2)))
    return grad_diff_bound / mu


def compute_source_influence_feature(
    snippets: List[EvidenceSnippet],
    P_t: np.ndarray,
    target_pair: Optional[Tuple[int, int]],
    rules: Optional[List[SoftLogicRule]] = None,
    mu: float = 0.05,
    alpha: float = 1.0,
) -> Tuple[float, int, float]:
    r"""
    计算式(3) 面向候选目标的来源影响特征:
      \chi_t(a) = p_{\ell h}(1 - p_{\ell h}) || z*(E_t, P_t^{\ell h <- 1}) - z*(E_t, P_t^{\ell h <- 0}) ||_2
    返回:
      (chi_val, iota_a, counterfactual_l2_diff)
    """
    _positive_mu(mu)
    validate_evidence(snippets, P_t)
    if target_pair is None:
        return 0.0, 0, 0.0
    ell, h = target_pair
    if ell == h or not (0 <= ell < len(snippets) and 0 <= h < len(snippets)):
        raise ValueError("target_pair must reference two distinct existing snippets")
    p_lh = float(P_t[ell, h])

    P_one = P_t.copy()
    P_one[ell, h] = 1.0
    P_one[h, ell] = 1.0

    P_zero = P_t.copy()
    P_zero[ell, h] = 0.0
    P_zero[h, ell] = 0.0

    z_one, _ = solve_convex_qp_inference(snippets, P_one, rules=rules, mu=mu, alpha=alpha)
    z_zero, _ = solve_convex_qp_inference(snippets, P_zero, rules=rules, mu=mu, alpha=alpha)

    cf_diff = float(np.linalg.norm(z_one - z_zero, ord=2))
    chi_val = p_lh * (1.0 - p_lh) * cf_diff
    return chi_val, 1, cf_diff


LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]


def terminal_calibrated_decision(
    z_star: np.ndarray,
    snippets: List[EvidenceSnippet],
    true_label: str,
    temperature: float = 0.35,
    prob_threshold: float = 0.75,
    sufficiency_threshold: float = 0.60,
) -> Dict[str, object]:
    """
    对应 2.2.4 节：证据充分性约束下的终端可信决策与 Chow 拒识损失计算。
    注意：logit 系数 (2.4, 2.6, 3.2, 2.2) 为预实验仿真手工设计的先验权重，
    非数据学习所得。正式实验 (E1-E4) 使用 ObservableTerminalVerifier 的训练分类器。
    四类标签:
      0: Supported
      1: Refuted
      2: Not Enough Evidence
      3: Conflicting Evidence/Cherrypicking
    损失定义 (Chow准则):
      正确判断 = 0.0, 错误判断 = 1.0, 系统暂缓判断 (Abstain) = 0.25
    门控条件:
      max_c p_c > 0.75 且 充分性评分 >= sufficiency_threshold 且 引文可回放校验通过
    """
    m = len(z_star) // 2
    z_pos = z_star[:m]
    z_neg = z_star[m:]

    mean_pos = float(np.mean(z_pos))
    max_neg = float(np.max(z_neg))
    mean_neg = float(np.mean(z_neg))
    conflict_score = float(np.mean(np.minimum(z_pos, z_neg)))
    unexplained_gap = float(np.mean(np.maximum(0.0, 1.0 - (z_pos + z_neg))))

    # 基于双通道状态计算四类原始 logit 得分
    logit_sup = 2.4 * mean_pos * (1.0 - max_neg) - 0.8 * max_neg
    logit_ref = 2.6 * (0.65 * max_neg + 0.35 * mean_neg) * (1.0 - 0.5 * conflict_score)
    logit_con = 3.2 * conflict_score + 0.8 * float(np.min([mean_pos, max_neg]))
    logit_nei = 2.2 * unexplained_gap - 0.6 * (mean_pos + max_neg)

    logits = np.array([logit_sup, logit_ref, logit_nei, logit_con], dtype=float)
    scaled = logits / max(temperature, 1e-4)
    scaled -= np.max(scaled)
    probs = np.exp(scaled) / np.sum(np.exp(scaled))

    pred_idx = int(np.argmax(probs))
    max_prob = float(probs[pred_idx])
    raw_pred_label = LABELS[pred_idx]

    # 证据充分性评分：综合关键事实覆盖度、图像出处完备性、双通道状态能量与核查器置信度
    fact_coverage = min(1.0, len(set(s.fact_id for s in snippets)) / max(1, m))
    has_provenance = 1.0 if any(s.is_image_provenance for s in snippets) else 0.35
    state_strength = float(np.clip(max(mean_pos, max_neg, conflict_score * 1.5), 0.0, 1.0))
    sufficiency_score = (
        0.35 * fact_coverage
        + 0.25 * has_provenance
        + 0.20 * state_strength
        + 0.20 * max_prob
    )
    citation_replay_ok = all(s.url_replayable for s in snippets) and len(snippets) > 0

    abstain = not (
        max_prob > prob_threshold
        and sufficiency_score >= sufficiency_threshold
        and citation_replay_ok
    )

    if abstain:
        final_decision = "ABSTAIN"
        chow_loss = 0.25
        is_correct = False
    else:
        final_decision = raw_pred_label
        is_correct = (raw_pred_label == true_label)
        chow_loss = 0.0 if is_correct else 1.0

    return {
        "probs": probs.tolist(),
        "raw_pred_label": raw_pred_label,
        "final_decision": final_decision,
        "max_prob": max_prob,
        "sufficiency_score": sufficiency_score,
        "citation_replay_ok": citation_replay_ok,
        "abstain": abstain,
        "chow_loss": chow_loss,
        "is_correct_non_abstain": is_correct,
    }
