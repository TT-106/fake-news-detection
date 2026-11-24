"""
式(1)—式(2)的数值性质与解析解检查。
不读取历史报告，也不把“本方法胜出”当作程序正确性的验收条件。
"""

import numpy as np

from src.core_models import (
    EvidenceSnippet,
    compute_discounted_weights,
    solve_convex_qp_inference,
    theoretical_stability_bound,
)


def test_eq1_same_source_discount_and_novel_fact_retention():
    """验证式(1)抑制同事实重复计票，同时完整保留同域名下的新增独立事实"""
    s0 = EvidenceSnippet(
        snippet_id=0, domain="nasa.gov", url="https://nasa.gov/1",
        fact_id=1, quality=np.array([0.9, 0.9]), r_pos=np.array([0.8, 0.1]), r_neg=np.array([0.1, 0.8])
    )
    # 同一域名 nasa.gov 下的另一条独立事实 (fact_id=2 != 1)
    s1 = EvidenceSnippet(
        snippet_id=1, domain="nasa.gov", url="https://nasa.gov/1#loc",
        fact_id=2, quality=np.array([0.8, 0.8]), r_pos=np.array([0.7, 0.1]), r_neg=np.array([0.1, 0.85])
    )
    # 另一网站转载 s0 的同一事实 (fact_id=1)，同源概率 0.9
    s2 = EvidenceSnippet(
        snippet_id=2, domain="repost.net", url="https://repost.net/copy",
        fact_id=1, quality=np.array([0.9, 0.9]), r_pos=np.array([0.8, 0.1]), r_neg=np.array([0.1, 0.8])
    )
    P = np.array([
        [1.0, 0.95, 0.90],
        [0.95, 1.0, 0.85],
        [0.90, 0.85, 1.0],
    ])
    W_prob = compute_discounted_weights([s0, s1, s2], P, mode="prob_discount")
    W_dom = compute_discounted_weights([s0, s1, s2], P, mode="domain_dedup")

    # s0 与 s2 同属 fact_id=1，故权重被折扣为 0.9 / (1 + 0.9)
    assert np.allclose(W_prob[:, 0], 0.9 / 1.9, atol=1e-6)
    assert np.allclose(W_prob[:, 2], 0.9 / 1.9, atol=1e-6)
    # s1 虽与 s0 同源/同域，但属于新增独立事实 fact_id=2，式(1) 保留其完整权重 0.8！
    assert np.allclose(W_prob[:, 1], 0.8, atol=1e-6)
    # 相反，粗粒度域名去重会把 s1 的权重完全置 0！
    assert np.allclose(W_dom[:, 1], 0.0, atol=1e-6)


def test_eq2_convex_qp_and_stability_bound():
    """验证式(2)凸二次规划求解器收敛性及强凸稳定性上界"""
    s0 = EvidenceSnippet(
        snippet_id=0, domain="a.org", url="u0", fact_id=1,
        quality=np.array([0.9, 0.85]), r_pos=np.array([0.85, 0.10]), r_neg=np.array([0.10, 0.90])
    )
    s1 = EvidenceSnippet(
        snippet_id=1, domain="b.org", url="u1", fact_id=1,
        quality=np.array([0.8, 0.8]), r_pos=np.array([0.80, 0.15]), r_neg=np.array([0.12, 0.85])
    )
    P1 = np.array([[1.0, 0.1], [0.1, 1.0]])
    P2 = np.array([[1.0, 0.8], [0.8, 1.0]])

    z1, _ = solve_convex_qp_inference([s0, s1], P1, mu=0.05)
    z2, _ = solve_convex_qp_inference([s0, s1], P2, mu=0.05)
    bound = theoretical_stability_bound([s0, s1], P1, P2, mu=0.05)
    actual_drift = float(np.linalg.norm(z1 - z2, ord=2))
    assert actual_drift <= bound + 1e-8




def test_solver_diagnostics_are_not_silently_ignored():
    s = EvidenceSnippet(0, "example.org", "https://example.org", 1,
                        np.array([0.8]), np.array([0.9]), np.array([0.1]))
    z, info = solve_convex_qp_inference([s], np.eye(1))
    assert info["solver_success"]
    assert info["projected_gradient_residual"] <= 1e-5
    np.testing.assert_allclose(z, [.8*.9/.85, .8*.1/.85], atol=1e-7)
