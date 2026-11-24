"""
预实验二：32组受控合成输入的双通道证据聚合与同源概率权重折扣数值仿真实验
对应申请书 1.4 节、2.2 节式(1)—式(3)、表2 (E1/E2) 及三(六)科研经历：
  采用 2^5 = 32 全因子受控实验设计 (Full-Factorial Controlled Design)，系统调控：
    - 因子 A (同源转载放大规模 copy_k): k=2 (轻度转载) vs k=5 (重度同源回声室放大)
    - 因子 B (同域名新增独立事实 same_domain_novel_fact): 无 vs 有 (检验粗粒度域名去重丢失新增事实的缺陷)
    - 因子 C (跨模态证据冲突结构 conflict_mode): 单向一致反驳 vs 旧图新用跨模态冲突 (画面真实但时空归属错配)
    - 因子 D (跨模态软逻辑规则约束 use_rules): 关闭 (alpha=0) vs 开启 (alpha=1.0, 蕴涵/必要条件/不相容三类规则)
    - 因子 E (来源概率估计噪声水平 noise_level): 低噪声 (sigma=0.06) vs 中高噪声 (sigma=0.18)

  在全部 32 组受控输入上定量对比 6 种证据赋权与聚合策略：
    1. independent (传统独立性假设，无同源折扣)
    2. domain_dedup (粗粒度按网站域名去重)
    3. max_quality_single (同事实簇仅保留单条最高质量记录)
    4. semantic_dedup (硬阈值去重 tau=0.5)
    5. prob_discount (本项目式(1) 同源概率软折扣，输入带噪预测矩阵 P_hat)
    6. oracle_source (人工标注真实同源矩阵上界 P_true)

  核心验证指标：
    - 相对真实理想状态 z_oracle 的状态恢复误差 ||z* - z_oracle||_2 (State Drift / RMSE)
    - 同源转载下的虚假支持度膨胀抑制率 (Echo-Chamber Inflation Suppression)
    - 同域名新增独立事实保留率 (Novel Fact Retention Rate)
    - 来源概率预测 Brier 分数及理论稳定性上界 (Theoretical Stability Bound) 100% 满足性检验
    - 式(3) 来源影响特征 chi_t(a) 对最有价值来源核验目标的排序命中率
"""

from __future__ import annotations
import itertools
import hashlib
import json
import os
from typing import Dict, List, Any, Tuple
import numpy as np

from src.core_models import (
    EvidenceSnippet,
    SoftLogicRule,
    compute_discounted_weights,
    solve_convex_qp_inference,
    theoretical_stability_bound,
    compute_source_influence_feature,
)


def paired_case_rng(copy_k: int, same_domain_novel_fact: bool, conflict_mode: str):
    """控制 use_rules/noise_sigma 比较时保持相同证据与标准化噪声抽样。"""
    key = f"20261107:{copy_k}:{int(same_domain_novel_fact)}:{conflict_mode}"
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return np.random.default_rng(seed)


def build_cross_modal_rules(m: int = 4) -> List[SoftLogicRule]:
    r"""
    构建 4 个原子声明上的三类跨模态软逻辑规则 (对应 2.2.2 节):
      声明 0 (j=0): 图像可见主体/场景真实性 (Visual Content Authenticity)
      声明 1 (j=1): 原始档案发布时间与配文声称时间吻合 (Temporal Alignment)
      声明 2 (j=2): 原始档案发生地点与配文声称地点吻合 (Spatial Alignment)
      声明 3 (j=3): 图文整体事件归属声明成立 (Overall Event Claim Validity)

    状态向量 z = [z_0^+, z_1^+, z_2^+, z_3^+, z_0^-, z_1^-, z_2^-, z_3^-]^T \in [0, 1]^8
    """
    assert m == 4
    rules: List[SoftLogicRule] = []

    # 规则1 (蕴涵 Implication): 若原始出处时间被反驳 (z_1^- 高)，则整体事件归属必被反驳 (z_3^- >= z_1^-)
    # 违反量: z_1^- - z_3^- > 0
    c1 = np.zeros(8)
    c1[4 + 1] = 1.0
    c1[4 + 3] = -1.0
    rules.append(
        SoftLogicRule(
            rule_id="R1_temporal_refute_implies_claim_refute",
            rule_type="implication",
            weight=1.2,
            c=c1,
            d=0.0,
            description="时间要素被反驳蕴涵整体图文事件声明被反驳 (z_1^- <= z_3^-)",
        )
    )

    # 规则2 (蕴涵 Implication): 若原始出处地点被反驳 (z_2^- 高)，则整体事件归属必被反驳 (z_3^- >= z_2^-)
    c2 = np.zeros(8)
    c2[4 + 2] = 1.0
    c2[4 + 3] = -1.0
    rules.append(
        SoftLogicRule(
            rule_id="R2_spatial_refute_implies_claim_refute",
            rule_type="implication",
            weight=1.2,
            c=c2,
            d=0.0,
            description="空间要素被反驳蕴涵整体图文事件声明被反驳 (z_2^- <= z_3^-)",
        )
    )

    # 规则3 (必要条件 Necessary Condition): 若整体事件声明受支持 (z_3^+ 高)，则时间与地点均需受支持
    # 违反量: z_3^+ - 0.5*(z_1^+ + z_2^+) > 0
    c3 = np.zeros(8)
    c3[3] = 1.0
    c3[1] = -0.5
    c3[2] = -0.5
    rules.append(
        SoftLogicRule(
            rule_id="R3_spatiotemporal_necessary_for_claim_support",
            rule_type="necessary",
            weight=1.0,
            c=c3,
            d=0.0,
            description="时空要素受支持是整体事件声明受支持的必要条件 (z_3^+ <= 0.5*(z_1^+ + z_2^+))",
        )
    )

    # 不设置同一声明正反通道的普遍互斥约束：申请书允许二者同时较高表示可信证据冲突。
    # 不相容规则必须有同一实体/事件/时空下的具体关系依据，不能为凑齐规则类型而生成。

    return rules


def generate_single_controlled_case(
    case_idx: int,
    copy_k: int,
    same_domain_novel_fact: bool,
    conflict_mode: str,
    use_rules: bool,
    noise_sigma: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    生成单组受控合成证据包 (m=4 个原子声明，n = 3 + copy_k 个证据片段)
    """
    m = 4
    snippets: List[EvidenceSnippet] = []

    # 片段 0: 权威图像原始出处档案 (如 NASA / AFP 首发档案)，核实声明0(画面内容真)并反驳声明1(时间错配)
    q0 = np.array([0.92, 0.95, 0.70, 0.88])
    if conflict_mode == "out_of_context":
        # 旧图新用：画面本身真实 (r_0^+ 高)，但原始拍摄时间与配文严重冲突 (r_1^- 高, r_3^- 高)
        r0_pos = np.array([0.90, 0.05, 0.15, 0.10])
        r0_neg = np.array([0.05, 0.94, 0.65, 0.88])
    else:
        # 一致反驳：画面与时间均直接指向反驳
        r0_pos = np.array([0.10, 0.06, 0.10, 0.08])
        r0_neg = np.array([0.85, 0.92, 0.80, 0.90])

    snippets.append(
        EvidenceSnippet(
            snippet_id=0,
            domain="nasa.gov",
            url="https://science.nasa.gov/earth-observatory/archive-1",
            fact_id=101,  # 规范事实 101: 原始拍摄时间档案事实
            quality=q0,
            r_pos=r0_pos,
            r_neg=r0_neg,
            is_image_provenance=True,
        )
    )

    # 片段 1: 补充空间/地点核查记录 (规范事实 102: 地点核验事实)
    # 当 same_domain_novel_fact=True 时，该独立新事实来自同一权威域名 nasa.gov（检验粗粒度域名去重是否误删！）
    dom_1 = "nasa.gov" if same_domain_novel_fact else "reuters.com"
    q1 = np.array([0.75, 0.65, 0.90, 0.84])
    r1_pos = np.array([0.82, 0.10, 0.08, 0.08]) if conflict_mode == "out_of_context" else np.array([0.12, 0.10, 0.08, 0.08])
    r1_neg = np.array([0.08, 0.60, 0.93, 0.86])
    snippets.append(
        EvidenceSnippet(
            snippet_id=1,
            domain=dom_1,
            url=f"https://{dom_1}/location-verification-record-2",
            fact_id=102,  # 规范事实 102 (独立新增事实！)
            quality=q1,
            r_pos=r1_pos,
            r_neg=r1_neg,
            is_image_provenance=False,
        )
    )

    # 片段 2: 独立第三方核查机构 (如 afp.com) 对规范事实 101 的互补多侧面核实
    # (片段0在声明0/1质量略高，片段2提供对声明1/3的强力独立反驳支撑；max_quality_single 会丢弃片段2导致支撑不足)
    q2 = np.array([0.82, 0.88, 0.68, 0.84])
    r2_pos = np.array([0.86, 0.06, 0.12, 0.06]) if conflict_mode == "out_of_context" else np.array([0.10, 0.06, 0.12, 0.06])
    r2_neg = np.array([0.06, 0.95, 0.84, 0.94])
    snippets.append(
        EvidenceSnippet(
            snippet_id=2,
            domain="factual.afp.com",
            url="https://factual.afp.com/independent-check",
            fact_id=101,  # 与片段0同属事实 101，系独立权威信源 (p=0.08)
            quality=q2,
            r_pos=r2_pos,
            r_neg=r2_neg,
            is_image_provenance=True,
        )
    )

    # 片段 3 .. 3+copy_k-1: 社交媒体/自媒体同源转载簇 (规范事实 201: 盲目转发误导性配文说法)
    for c_idx in range(copy_k):
        q_copy = np.clip(
            np.array([0.62, 0.58, 0.58, 0.65]) - 0.02 * c_idx + rng.normal(0, 0.015, size=m),
            0.35,
            0.75,
        )
        r_copy_pos = np.clip(
            np.array([0.88, 0.84, 0.82, 0.89]) + rng.normal(0, 0.015, size=m),
            0.65,
            0.96,
        )
        r_copy_neg = np.array([0.05, 0.06, 0.06, 0.05])
        snippets.append(
            EvidenceSnippet(
                snippet_id=3 + c_idx,
                domain=f"viral-repost-{c_idx + 1}.net",
                url=f"https://viral-repost-{c_idx + 1}.net/post/{case_idx}_{c_idx}",
                fact_id=201,  # 规范事实 201: 误导性配文说法
                quality=q_copy,
                r_pos=r_copy_pos,
                r_neg=r_copy_neg,
                is_image_provenance=False,
            )
        )

    n = len(snippets)
    # 构造真实同源关系矩阵 P_true：
    # - 片段 (0, 2) 为独立权威核查机构 (真实同源概率 0.08)
    # - 转载簇 3..n-1 内部包含直接复制 (0.92) 与改写洗稿 (0.58，靠近硬阈值边界 0.50)
    P_true = np.eye(n, dtype=float)
    P_true[0, 2] = P_true[2, 0] = 0.08
    for i in range(3, n):
        for j in range(i + 1, n):
            val_true = 0.92 if ((i + j) % 2 == 1) else 0.58
            P_true[i, j] = val_true
            P_true[j, i] = val_true

    # 构造带估计误差的预测同源概率矩阵 P_hat：
    # 当存在边界改写同源 (0.58) 时，硬阈值去重 (tau=0.5) 极易发生 0/1 阶跃截断失真，
    # 而式(1) 连续概率折扣能平滑保持权重稳定并保留 (0, 2) 的双源佐证
    P_hat = P_true.copy()
    for i in range(n):
        for j in range(i + 1, n):
            base_p = P_true[i, j]
            if abs(base_p - 0.58) < 0.05:
                val = float(np.clip(base_p - 0.12 + rng.normal(0, 0.35 * noise_sigma), 0.38, 0.49))
            elif base_p > 0.8:
                val = float(np.clip(base_p - abs(rng.normal(0.35 * noise_sigma, 0.35 * noise_sigma)), 0.74, 0.98))
            else:
                val = float(np.clip(base_p + abs(rng.normal(0.35 * noise_sigma, 0.25 * noise_sigma)), 0.02, 0.24))
            P_hat[i, j] = val
            P_hat[j, i] = val

    # 专门设置一对高不确定性待核验同源对 (片段 3 与 片段 4)，使 p_{3,4} 处于 ~0.55 用于检验式(3)来源影响特征
    P_uncertain = P_hat.copy()
    if n >= 5:
        P_uncertain[3, 4] = 0.54
        P_uncertain[4, 3] = 0.54

    rules = build_cross_modal_rules(m=m) if use_rules else []
    alpha_val = 1.0 if use_rules else 0.0
    mu_val = 0.05

    # 计算上界基准状态 z_oracle (使用真实 P_true + 软逻辑规则开启)
    full_rules = build_cross_modal_rules(m=m)
    z_oracle, _ = solve_convex_qp_inference(
        snippets, P_true, rules=full_rules, mu=mu_val, alpha=1.0, weight_mode="prob_discount"
    )
    # 同样计算当前规则配置下的 P_true 状态 z_true_same_rule (专用于检验纯来源误差稳定性上界)
    z_true_same_rule, _ = solve_convex_qp_inference(
        snippets, P_true, rules=rules, mu=mu_val, alpha=alpha_val, weight_mode="prob_discount"
    )

    methods = [
        "independent",
        "domain_dedup",
        "max_quality_single",
        "semantic_dedup",
        "prob_discount",
        "oracle_source",
    ]
    method_results: Dict[str, Any] = {}

    for mode in methods:
        P_used = P_true if mode == "oracle_source" else P_hat
        w_mode = "prob_discount" if mode == "oracle_source" else mode

        W = compute_discounted_weights(snippets, P_used, mode=w_mode, hard_threshold=0.5)
        z_star, info = solve_convex_qp_inference(
            snippets, P_used, rules=rules, mu=mu_val, alpha=alpha_val, weight_mode=w_mode
        )

        # 指标1: 相对全局理想状态 z_oracle 的 L2 状态漂移
        drift_vs_oracle = float(np.linalg.norm(z_star - z_oracle, ord=2))
        # 指标2: 相对同规则 P_true 状态 z_true_same_rule 的纯来源误差漂移
        drift_vs_true_source = float(np.linalg.norm(z_star - z_true_same_rule, ord=2))
        # 指标3: 声明3(整体事件)上的虚假支持度 z_3^+ 与正确反驳度 z_3^- 之差 (越小说明越不受回声室误导)
        claim3_support_inflation = float(z_star[3])
        claim3_refute_strength = float(z_star[4 + 3])
        correct_refute_margin = float(z_star[4 + 3] - z_star[3])

        # 指标4: 新增独立事实保留率 (检验权威规范事实 101 与 102 是否均获得非零有效权重)
        fact101_weight = float(np.sum([np.mean(W[:, i]) for i, s in enumerate(snippets) if s.fact_id == 101]))
        fact102_weight = float(np.sum([np.mean(W[:, i]) for i, s in enumerate(snippets) if s.fact_id == 102]))
        novel_fact_retained = 1.0 if (fact101_weight > 1e-6 and fact102_weight > 1e-6) else (0.5 if (fact101_weight > 1e-6 or fact102_weight > 1e-6) else 0.0)

        method_results[mode] = {
            "drift_vs_oracle_l2": round(drift_vs_oracle, 5),
            "drift_vs_true_source_l2": round(drift_vs_true_source, 5),
            "claim3_support_z_pos": round(claim3_support_inflation, 4),
            "claim3_refute_z_neg": round(claim3_refute_strength, 4),
            "correct_refute_margin": round(correct_refute_margin, 4),
            "novel_fact_retention_rate": novel_fact_retained,
            "z_star": [round(float(x), 4) for x in z_star],
        }

    # 计算非对角元素上的 Brier Score: 2/(n(n-1)) \sum_{i<j} (P_hat_{ij} - P_true_{ij})^2
    off_diag_diffs = [
        (P_hat[i, j] - P_true[i, j]) ** 2
        for i in range(n)
        for j in range(i + 1, n)
    ]
    brier_score = float(np.mean(off_diag_diffs))

    # 验证理论稳定性上界: ||z*(E, P_hat) - z*(E, P_true)||_2 <= Bound(P_hat, P_true)
    actual_source_drift = method_results["prob_discount"]["drift_vs_true_source_l2"]
    theo_bound = theoretical_stability_bound(snippets, P_hat, P_true, mu=mu_val)
    bound_satisfied = bool(actual_source_drift <= theo_bound + 1e-9)

    # 验证式(3) 来源影响特征: 比较核验同源转载对 (3,4) 与无关独立对 (0,1) 的 chi_t(a)
    chi_repost_pair, _, cf_diff_repost = compute_source_influence_feature(
        snippets, P_uncertain, (3, 4), rules=rules, mu=mu_val, alpha=alpha_val
    )
    chi_indep_pair, _, cf_diff_indep = compute_source_influence_feature(
        snippets, P_uncertain, (0, 1), rules=rules, mu=mu_val, alpha=alpha_val
    )

    return {
        "case_id": f"SYN_{case_idx:02d}",
        "factors": {
            "copy_k": copy_k,
            "same_domain_novel_fact": same_domain_novel_fact,
            "conflict_mode": conflict_mode,
            "use_rules": use_rules,
            "noise_sigma": noise_sigma,
        },
        "brier_score": round(brier_score, 5),
        "stability_verification": {
            "actual_state_drift_l2": round(actual_source_drift, 5),
            "theoretical_upper_bound": round(theo_bound, 5),
            "tightness_ratio": round(actual_source_drift / max(theo_bound, 1e-9), 4),
            "bound_satisfied": bound_satisfied,
        },
        "source_influence_feature_check": {
            "chi_high_impact_repost_pair": round(chi_repost_pair, 5),
            "chi_low_impact_indep_pair": round(chi_indep_pair, 5),
            "correctly_ranks_high_impact_pair": bool(chi_repost_pair > chi_indep_pair),
        },
        "methods": method_results,
    }


def run_exp2(output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(20261107)

    # 2^5 = 32 组全因子实验配置
    factor_grid = list(
        itertools.product(
            [2, 5],                          # copy_k: 轻度 vs 重度同源转载
            [False, True],                   # same_domain_novel_fact: 异域 vs 同域多独立事实
            ["unanimous_refute", "out_of_context"],  # conflict_mode
            [False, True],                   # use_rules: 软逻辑规则消融 vs 开启
            [0.06, 0.18],                    # noise_sigma: 低噪声 vs 中高噪声
        )
    )
    assert len(factor_grid) == 32

    cases = []
    for idx, (copy_k, same_dom, c_mode, u_rules, n_sigma) in enumerate(factor_grid, start=1):
        case_res = generate_single_controlled_case(
            case_idx=idx,
            copy_k=copy_k,
            same_domain_novel_fact=same_dom,
            conflict_mode=c_mode,
            use_rules=u_rules,
            noise_sigma=n_sigma,
            rng=paired_case_rng(copy_k, same_dom, c_mode),
        )
        cases.append(case_res)

    # 汇总 6 种方法在 32 组实验上的总体均值与标准差
    methods = [
        "independent",
        "domain_dedup",
        "max_quality_single",
        "semantic_dedup",
        "prob_discount",
        "oracle_source",
    ]
    summary_by_method: Dict[str, Any] = {}
    for m_name in methods:
        drifts = [c["methods"][m_name]["drift_vs_oracle_l2"] for c in cases]
        src_drifts = [c["methods"][m_name]["drift_vs_true_source_l2"] for c in cases]
        inflations = [c["methods"][m_name]["claim3_support_z_pos"] for c in cases]
        margins = [c["methods"][m_name]["correct_refute_margin"] for c in cases]
        retentions = [c["methods"][m_name]["novel_fact_retention_rate"] for c in cases]
        # 仅在 same_domain_novel_fact=True 的 16 组上统计同域新增事实保留率
        same_dom_retentions = [
            c["methods"][m_name]["novel_fact_retention_rate"]
            for c in cases
            if c["factors"]["same_domain_novel_fact"]
        ]

        summary_by_method[m_name] = {
            "mean_state_drift_vs_oracle_l2": round(float(np.mean(drifts)), 4),
            "std_state_drift_vs_oracle_l2": round(float(np.std(drifts)), 4),
            "mean_source_error_drift_l2": round(float(np.mean(src_drifts)), 4),
            "mean_echo_chamber_false_support_z_pos": round(float(np.mean(inflations)), 4),
            "mean_correct_refute_margin": round(float(np.mean(margins)), 4),
            "overall_novel_fact_retention_rate": round(float(np.mean(retentions)), 4),
            "same_domain_novel_fact_retention_rate": round(float(np.mean(same_dom_retentions)), 4),
        }

    # 统计软逻辑规则开启 (16组) vs 关闭 (16组) 在 prob_discount 下的改善
    rules_on_drifts = [
        c["methods"]["prob_discount"]["drift_vs_oracle_l2"]
        for c in cases if c["factors"]["use_rules"]
    ]
    rules_off_drifts = [
        c["methods"]["prob_discount"]["drift_vs_oracle_l2"]
        for c in cases if not c["factors"]["use_rules"]
    ]

    brier_low_noise = [c["brier_score"] for c in cases if c["factors"]["noise_sigma"] == 0.06]
    brier_high_noise = [c["brier_score"] for c in cases if c["factors"]["noise_sigma"] == 0.18]

    all_bounds_ok = all(c["stability_verification"]["bound_satisfied"] for c in cases)
    chi_ranking_acc = float(
        np.mean([1.0 if c["source_influence_feature_check"]["correctly_ranks_high_impact_pair"] else 0.0 for c in cases])
    )

    report = {
        "experiment": "Exp2_32_Controlled_Synthetic_Input_Numerical_Simulations",
        "evidence_protocol": "CONTROLLED_SYNTHETIC_MECHANISM_ONLY",
        "formal_multimodal_results": False,
        "num_controlled_configurations": len(cases),
        "factorial_design": "2^5 = 32 Full Factorial (copy_k x same_domain_novel_fact x conflict_mode x use_rules x noise_sigma)",
        "summary_by_method": summary_by_method,
        "ablation_and_theoretical_checks": {
            "soft_logic_rules_enabled_mean_drift": round(float(np.mean(rules_on_drifts)), 4),
            "soft_logic_rules_disabled_mean_drift": round(float(np.mean(rules_off_drifts)), 4),
            "relative_drift_reduction_by_rules": round(
                float((np.mean(rules_off_drifts) - np.mean(rules_on_drifts)) / max(1e-9, np.mean(rules_off_drifts))),
                4,
            ),
            "brier_score_low_noise_mean": round(float(np.mean(brier_low_noise)), 5),
            "brier_score_high_noise_mean": round(float(np.mean(brier_high_noise)), 5),
            "theoretical_stability_bound_100pct_satisfied": all_bounds_ok,
            "max_actual_to_bound_ratio": round(
                max(c["stability_verification"]["tightness_ratio"] for c in cases), 4
            ),
            "source_influence_chi_ranking_hit_rate": round(chi_ranking_acc, 4),
        },
        "cases": cases,
    }

    with open(os.path.join(output_dir, "exp2_synthetic_32_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    rep = run_exp2()
    print("Exp2 (32 controlled synthetic simulations) finished successfully.")
    for k, v in rep["summary_by_method"].items():
        print(f"  {k:20s}: drift={v['mean_state_drift_vs_oracle_l2']:.4f}, margin={v['mean_correct_refute_margin']:.4f}, same_dom_retention={v['same_domain_novel_fact_retention_rate']:.2f}")
