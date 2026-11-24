r"""
第一阶段：实验审计与数据划分/合成循环性修复 (Phase 1: Experimental Audit & Remediation)
严格执行以下 6 项审计：
  1. 检查数据划分、事件及图像 (CRC32) 泄漏、训练/验证隔离，输出 v1 泄漏清单并生成零泄漏 v2 划分
  2. 核实 v1 中来源关系 374/530 对的来源，量化证明 v1 的 100% 准确率由目标条件特征生成泄漏造成
  3. 复算 v1 所有实验指标、相对改善与 Bootstrap 置信区间，核查四舍五入与未舍入差异
  4. 检查 v1 合成数据 (Exp2) 的理想状态 z_oracle 是否直接依赖待验证软逻辑规则 (alpha=1.0)，修复为独立外部真值状态 z_gt
  5. 检查 v1 费用记录、候选行动和模型输入是否包含未来/真值标签信息 (label -> r_pos/r_neg, chi -> continuous_gain)
  6. 输出完整问题清单 `outputs/phase1_audit_report.json` 并生成修复后的 `outputs/averimatec_event_isolated_splits_v2.json`
"""

from __future__ import annotations
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any
import numpy as np

from src.exp1_averimatec_audit_and_split import (
    extract_clean_domain,
    normalize_url_for_matching,
    tokenize_entities,
)


def audit_and_rebuild_splits(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    crc_map: Dict[str, Dict[str, Any]],
    v1_splits_data: Dict[str, Any],
    seed: int = 20261107,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    审计 v1 数据划分中的事件与图像 CRC32 泄漏，并构建零泄漏的 v2 事件簇隔离划分。
    """
    n_train = len(train_samples)
    n_val = len(val_samples)

    v1_splits = v1_splits_data["splits"]
    v1_idx_to_split = {
        i: s_name for s_name, s_info in v1_splits.items() for i in s_info["indices"]
    }

    # 提取每条样本的全部图像 (含 claim_images 与 questions[*].input_images) 的真实内容指纹 (crc32, uncomp_size)
    def get_sample_image_fingerprints(sample: Dict[str, Any]) -> Set[Tuple[str, int]]:
        fps: Set[Tuple[str, int]] = set()
        imgs = list(sample.get("claim_images") or [])
        for q in (sample.get("questions") or []):
            imgs.extend(q.get("input_images") or [])
        for im in imgs:
            base_im = os.path.basename(im)
            if base_im in crc_map:
                fps.add((crc_map[base_im]["crc32"], int(crc_map[base_im]["uncomp_size"])))
            else:
                fps.add((base_im, -1))
        return fps

    train_fps = [get_sample_image_fingerprints(s) for s in train_samples]
    val_fps = [get_sample_image_fingerprints(s) for s in val_samples]

    # 1. 检查 train (793) 与 val (152) 之间的图像、文章链接、原始声明链接与声明文本泄漏
    all_train_crc = {fp for s_fp in train_fps for fp in s_fp}
    all_val_crc = {fp for s_fp in val_fps for fp in s_fp}
    shared_train_val_crc = all_train_crc & all_val_crc

    train_articles = {normalize_url_for_matching(s.get("article") or "") for s in train_samples} - {""}
    val_articles = {normalize_url_for_matching(s.get("article") or "") for s in val_samples} - {""}
    shared_train_val_articles = train_articles & val_articles

    train_orig_urls = {
        normalize_url_for_matching((s.get("metadata") or {}).get("original_claim_url") or "")
        for s in train_samples
    } - {""}
    val_orig_urls = {
        normalize_url_for_matching((s.get("metadata") or {}).get("original_claim_url") or "")
        for s in val_samples
    } - {""}
    shared_train_val_orig_urls = train_orig_urls & val_orig_urls

    # 2. 检查 v1 划分内部跨子集的图像 CRC32 泄漏
    crc_to_train_idxs: Dict[Tuple[str, int], Set[int]] = defaultdict(set)
    for idx, fps in enumerate(train_fps):
        for fp in fps:
            crc_to_train_idxs[fp].add(idx)

    v1_leaked_image_groups = []
    v1_within_split_unmerged_image_groups = []
    for fp, idxs in crc_to_train_idxs.items():
        if len(idxs) >= 2:
            sps = sorted({v1_idx_to_split[i] for i in idxs})
            if len(sps) > 1:
                v1_leaked_image_groups.append({
                    "image_crc32": fp[0],
                    "uncomp_bytes": fp[1],
                    "train_indices": sorted(idxs),
                    "leaked_across_splits": sps,
                })
            else:
                v1_within_split_unmerged_image_groups.append({
                    "image_crc32": fp[0],
                    "uncomp_bytes": fp[1],
                    "train_indices": sorted(idxs),
                    "split": sps[0],
                })

    # 3. 构建修复后的 v2 事件簇图：合并所有共享图像 CRC32、无上限共享具体证据 URL、共享声明/文章链接及高重合实体词元的样本
    parent = list(range(n_train))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # (a) 合并共享任意图像 CRC32 的训练样本
    for fp, idxs in crc_to_train_idxs.items():
        idx_list = sorted(idxs)
        for other in idx_list[1:]:
            union(idx_list[0], other)

    # (b) 合并共享声明文本、核查文章 URL、原始声明 URL、图像前缀及任意具体证据 URL (取消 <=8 限制)
    text_map: Dict[str, int] = {}
    article_map: Dict[str, int] = {}
    orig_url_map: Dict[str, int] = {}
    img_prefix_map: Dict[str, int] = {}
    evidence_url_map: Dict[str, List[int]] = defaultdict(list)
    sample_tokens: List[Set[str]] = []

    for idx, s in enumerate(train_samples):
        ct = re.sub(r"\s+", " ", (s.get("claim_text") or "").strip().lower())
        if ct:
            if ct in text_map:
                union(idx, text_map[ct])
            else:
                text_map[ct] = idx

        art = normalize_url_for_matching(s.get("article") or "")
        if art:
            if art in article_map:
                union(idx, article_map[art])
            else:
                article_map[art] = idx

        meta = s.get("metadata") or {}
        orig_u = normalize_url_for_matching(meta.get("original_claim_url") or "")
        if orig_u:
            if orig_u in orig_url_map:
                union(idx, orig_url_map[orig_u])
            else:
                orig_url_map[orig_u] = idx

        for img in (s.get("claim_images") or []):
            prefix = img.split("#")[0]
            if prefix:
                if prefix in img_prefix_map:
                    union(idx, img_prefix_map[prefix])
                else:
                    img_prefix_map[prefix] = idx

        for q in (s.get("questions") or []):
            for ans in (q.get("answers") or []):
                u = normalize_url_for_matching(ans.get("source_url") or "")
                if u and u != "metadata" and len(u) > 25:
                    evidence_url_map[u].append(idx)

        sample_tokens.append(tokenize_entities(s.get("claim_text") or ""))

    for u, idxs in evidence_url_map.items():
        if len(idxs) >= 2:
            first = idxs[0]
            for other in idxs[1:]:
                union(first, other)

    for i in range(n_train):
        ti = sample_tokens[i]
        if len(ti) < 3:
            continue
        for j in range(i + 1, n_train):
            tj = sample_tokens[j]
            if len(tj) < 3:
                continue
            inter = len(ti & tj)
            if inter >= 4 and (inter / len(ti | tj)) >= 0.55:
                union(i, j)

    clusters_dict: Dict[int, List[int]] = defaultdict(list)
    for i in range(n_train):
        clusters_dict[find(i)].append(i)

    clusters = list(clusters_dict.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)

    targets = {
        "pilot_probe": 20,
        "dev_calibration": 60,
        "mechanism_val": 60,
        "action_train": 180,
        "reserve_pool": 473,
    }
    splits_v2: Dict[str, List[int]] = {k: [] for k in targets}

    minority_clusters = [
        c for c in clusters
        if any(train_samples[i].get("label") != "Refuted" for i in c)
    ]
    majority_clusters = [
        c for c in clusters
        if all(train_samples[i].get("label") == "Refuted" for i in c)
    ]

    # 确保 dev_calibration、mechanism_val、action_train、pilot_probe 均均衡覆盖少数类簇
    split_order = ["dev_calibration", "mechanism_val", "action_train", "pilot_probe", "reserve_pool"]
    for idx_c, c in enumerate(minority_clusters):
        placed = False
        for s_name in split_order[idx_c % len(split_order):] + split_order[:idx_c % len(split_order)]:
            if len(splits_v2[s_name]) + len(c) <= targets[s_name]:
                splits_v2[s_name].extend(c)
                placed = True
                break
        if not placed:
            splits_v2["reserve_pool"].extend(c)

    multi_maj = [c for c in majority_clusters if len(c) >= 2]
    single_maj = [c for c in majority_clusters if len(c) == 1]

    for c in multi_maj:
        for s_name in ["action_train", "dev_calibration", "mechanism_val", "pilot_probe", "reserve_pool"]:
            if len(splits_v2[s_name]) + len(c) <= targets[s_name]:
                splits_v2[s_name].extend(c)
                break
        else:
            splits_v2["reserve_pool"].extend(c)

    for c in single_maj:
        for s_name in ["pilot_probe", "dev_calibration", "mechanism_val", "action_train", "reserve_pool"]:
            if len(splits_v2[s_name]) + len(c) <= targets[s_name]:
                splits_v2[s_name].extend(c)
                break

    # 严格验证 v2 的跨集事件簇泄漏 == 0 且跨集图像 CRC32 泄漏 == 0
    v2_idx_to_split = {i: s_name for s_name, idxs in splits_v2.items() for i in idxs}
    v2_cluster_leakage = 0
    assigned_cid: Dict[int, str] = {}
    for s_name, idxs in splits_v2.items():
        assert len(idxs) == targets[s_name]
        for i in idxs:
            cid = find(i)
            if cid in assigned_cid and assigned_cid[cid] != s_name:
                v2_cluster_leakage += 1
            assigned_cid[cid] = s_name

    v2_crc_leakage = 0
    for fp, idxs in crc_to_train_idxs.items():
        if len({v2_idx_to_split[i] for i in idxs}) > 1:
            v2_crc_leakage += 1

    assert v2_cluster_leakage == 0 and v2_crc_leakage == 0

    split_summary_v2 = {}
    for s_name, idxs in splits_v2.items():
        lbls = Counter(train_samples[i].get("label") for i in idxs)
        split_summary_v2[s_name] = {
            "count": len(idxs),
            "num_event_clusters": len({find(i) for i in idxs}),
            "label_distribution": dict(lbls),
            "indices": sorted(idxs),
        }

    v2_split_obj = {
        "version": "v2_crc32_and_url_leak_free",
        "total_train_samples": n_train,
        "total_event_clusters": len(clusters),
        "multi_sample_event_clusters": sum(1 for c in clusters if len(c) >= 2),
        "max_cluster_size": max(len(c) for c in clusters),
        "cross_split_cluster_leakage": v2_cluster_leakage,
        "cross_split_image_crc32_leakage": v2_crc_leakage,
        "splits": split_summary_v2,
    }

    audit_split_section = {
        "train_val_isolation_check": {
            "shared_image_crc32_between_train793_and_val152": len(shared_train_val_crc),
            "shared_factcheck_articles_between_train793_and_val152": len(shared_train_val_articles),
            "shared_original_claim_urls_between_train793_and_val152": len(shared_train_val_orig_urls),
            "is_val152_strictly_isolated_from_train793": (
                len(shared_train_val_crc) == 0
                and len(shared_train_val_articles) == 0
                and len(shared_train_val_orig_urls) == 0
            ),
        },
        "v1_internal_split_issues_found": {
            "root_cause": (
                "v1仅按图像文件名前缀 img.split('#')[0] (即样本自身MongoDB ID) 合并图像，"
                "未读取 images.zip 的真实文件 CRC32 校验和，且忽略了 questions[*].input_images。"
            ),
            "shared_image_crc32_groups_in_train793": len([k for k, v in crc_to_train_idxs.items() if len(v) >= 2]),
            "v1_cross_split_leaked_image_crc32_groups_count": len(v1_leaked_image_groups),
            "v1_cross_split_leaked_image_crc32_details": v1_leaked_image_groups,
            "v1_within_split_unmerged_image_crc32_groups_count": len(v1_within_split_unmerged_image_groups),
        },
        "v2_remediation_verification": {
            "v1_total_event_clusters": v1_splits_data["total_event_clusters"],
            "v2_total_event_clusters": len(clusters),
            "v2_multi_sample_clusters": sum(1 for c in clusters if len(c) >= 2),
            "v2_cross_split_cluster_leakage": v2_cluster_leakage,
            "v2_cross_split_image_crc32_leakage": v2_crc_leakage,
        },
    }

    return audit_split_section, v2_split_obj


def audit_v1_source_pairs_and_metrics(output_dir: str = "outputs") -> Dict[str, Any]:
    """
    审计第 2、3、4、5 项：
      - 核实 v1 中 374/530 来源关系对的组成与 100% 准确率的目标泄漏根因
      - 复算 v1 中所有相对改善指标与 Bootstrap 置信区间
      - 审计 v1 Exp2 合成数据中 z_oracle 对待验证软逻辑规则 (alpha=1.0) 的循环依赖
      - 审计 v1 Exp3 中模型输入对真实标签 label 的泄漏及行动收益对 chi_val 的自循环构造
    """
    with open(os.path.join(output_dir, "archive_v1_original/exp2_synthetic_32_report.json"), encoding="utf-8") as f:
        v1_exp2 = json.load(f)
    with open(os.path.join(output_dir, "archive_v1_original/exp3_pilot_and_action_gain_report.json"), encoding="utf-8") as f:
        v1_exp3 = json.load(f)

    # 复算 v1 Exp2 相对改善
    ind_drift = v1_exp2["summary_by_method"]["independent"]["mean_state_drift_vs_oracle_l2"]
    prob_drift = v1_exp2["summary_by_method"]["prob_discount"]["mean_state_drift_vs_oracle_l2"]
    recomputed_exp2_rel_red = (ind_drift - prob_drift) / ind_drift

    # 检验 v1 Exp2 中 oracle_source 在 use_rules=False (16组) 与 use_rules=True (16组) 下的漂移：
    # 证明 v1 的 z_oracle 直接绑定了 use_rules=True (alpha=1.0)
    oracle_drifts_rules_on = [
        c["methods"]["oracle_source"]["drift_vs_oracle_l2"]
        for c in v1_exp2["cases"] if c["factors"]["use_rules"]
    ]
    oracle_drifts_rules_off = [
        c["methods"]["oracle_source"]["drift_vs_oracle_l2"]
        for c in v1_exp2["cases"] if not c["factors"]["use_rules"]
    ]

    # 复算 v1 Exp3 相对改善
    reg_full = v1_exp3["e3_common_history_action_gain_evaluation"]["methods"]["full_with_source_influence_chi"]["simple_regret_mean"]
    reg_no = v1_exp3["e3_common_history_action_gain_evaluation"]["methods"]["ablation_without_source_influence_chi"]["simple_regret_mean"]
    recomputed_from_rounded = (reg_no - reg_full) / reg_no
    reported_in_v1 = v1_exp3["e3_common_history_action_gain_evaluation"]["hypothesis_test_chi_contribution"]["relative_simple_regret_reduction_rate"]

    return {
        "audit_2_source_pair_374_530_and_100pct_accuracy": {
            "origin_of_374_and_530_pairs": (
                "v1 在 extract_sample_evidence_and_source_pairs 中将每条声明不足4条的原始问答用合成记录补齐至4~8条，"
                "60条 dev_calibration 产生 418 对候选（剔除44对无法核实态后剩 374 对训练对），"
                "20条 pilot_probe + 60条 mechanism_val 产生 592 对候选（剔除62对无法核实态后剩 530 对评估对）。"
            ),
            "cause_of_100pct_accuracy_and_0002_brier": (
                "严重特征泄漏（Target-Conditioned Feature Generation）：v1 第273–288行在判定 y_rel=1 或 y_rel=0 后，"
                "直接按 y_rel 分支从互不重叠的区间 [0.60, 1.00] 与 [0.00, 0.35] 采样生成四维特征 f_vec，"
                "导致正负样本在特征空间线性完全可分，从而产生虚假的 100.0% 准确率与 0.0020 Brier 分数。"
            ),
            "remediation": (
                "移除所有基于 y_rel 的条件随机数生成与合成补齐 URL，改为纯观测特征提取："
                "(1) 规范化URL路径前缀与域名链匹配度；(2) URL/文本年份日期戳接近度；"
                "(3) 媒体类型与图像CRC/检索方式重合度；(4) 证据文本字符3-gram与词元TF-IDF余弦相似度。"
            ),
        },
        "audit_3_metric_and_ci_recomputation": {
            "v1_exp2_drift_reduction_prob_vs_independent_recomputed": round(recomputed_exp2_rel_red, 4),
            "v1_exp3_simple_regret_full": reg_full,
            "v1_exp3_simple_regret_no_chi": reg_no,
            "v1_exp3_relative_reduction_reported_unrounded": reported_in_v1,
            "v1_exp3_relative_reduction_from_rounded_table": round(recomputed_from_rounded, 4),
            "rounding_discrepancy_note": (
                f"未舍入浮点数计算为 {reported_in_v1*100:.2f}%，按保留5位小数表格值复算为 {recomputed_from_rounded*100:.2f}%（相差 0.05 个百分点）。"
                "v2 中同时报告精确浮点值与表格复算值以消除舍入歧义。"
            ),
        },
        "audit_4_synthetic_exp2_oracle_circularity": {
            "circularity_detected": True,
            "evidence_of_circularity": {
                "oracle_source_mean_drift_when_use_rules_True": round(float(np.mean(oracle_drifts_rules_on)), 5),
                "oracle_source_mean_drift_when_use_rules_False": round(float(np.mean(oracle_drifts_rules_off)), 5),
                "explanation": (
                    "v1 Exp2 将理想状态 z_oracle 定义为 solve_convex_qp_inference(snippets, P_true, rules=full_rules, alpha=1.0, weight_mode='prob_discount')。"
                    "这导致：(1) 当测试 use_rules=False (alpha=0) 时，即便给予定律上界 P_true (oracle_source)，其与 z_oracle 的漂移也高达 0.3376，"
                    "使『开启软逻辑规则降低 95.74% 漂移』退化为用 alpha=1.0 的输出检验 alpha=1.0 的同义反复；"
                    "(2) z_oracle 内嵌了 prob_discount 的加权形式。"
                ),
            },
            "remediation": (
                "在修复后的 Exp2 (v2) 中，定义完全独立于求解器、权重模式与软逻辑规则的外部真实原子状态向量 z_gt in {0, 1}^{2m} "
                "(例如旧图新用场景下真实状态为 z_gt^+=[1,0,0,0], z_gt^-=[0,1,1,1]；一致反驳场景下为 z_gt^+=[0,0,0,0], z_gt^-=[1,1,1,1])，"
                "分别独立评估：(A) 相对外部真值 z_gt 的状态恢复误差 ||z* - z_gt||_2；"
                "(B) 固定相同规则参数 alpha 下纯来源估计误差导致的状态扰动 ||z*(E, P_hat) - z*(E, P_true)||_2。"
            ),
        },
        "audit_5_future_and_label_leakage_in_inputs_and_actions": {
            "leakage_items_found": [
                {
                    "id": "L1_ground_truth_label_in_stance_scores",
                    "location": "src/exp3_pilot_20_active_verification.py:L231-L242",
                    "description": (
                        "v1 在构造证据片段的直接支持/反驳评分 (r_pos, r_neg) 时直接读取了样本的真实标签 sample['label'] "
                        "(例如 if label == 'Refuted': r_neg ~ Uniform(0.72, 0.93))，属于严重的未来真值标签泄漏。"
                    ),
                },
                {
                    "id": "L2_self_fulfilling_chi_in_action_gain_target",
                    "location": "src/exp3_pilot_20_active_verification.py:L464",
                    "description": (
                        "v1 在模拟候选行动 a2 (来源核验) 的真实收益 continuous_gain 时，直接使用了公式 "
                        "`continuous_gain = 4.6 * chi_val + 0.10 * delta_loss - 0.005 + noise`，"
                        "把输入特征 chi_val 线性硬编码进监督目标 y_gain，导致包含 chi_val 的回归器获得人为放大的预测优势（69.79% 遗憾下降）。"
                    ),
                },
                {
                    "id": "L3_synthetic_regime_coin_flip_in_rollout",
                    "location": "src/exp3_pilot_20_active_verification.py:L353-L390",
                    "description": (
                        "v1 使用随机数 regime_u < 0.46 人为篡改初始证据片段的立场与质量，而非基于真实检索证据池中的多源转载与冲突记录。"
                    ),
                },
            ],
            "remediation": (
                "在 v2 真实实验流水线 (src/real_averimatec_pipeline.py) 中彻底废除上述三处泄漏："
                "(1) 严禁任何特征提取器、立场打分器、质量评估器或行动策略读取 label / justification / refuting_reasons / image_misuse_types；"
                "(2) 仅利用声明文本、日期、图像CRC/元数据及检索问答文本的可观测词汇、年份冲突、布尔回答、Unanswerable标记，在 dev_calibration (60条) 上训练立场与终端分类器；"
                "(3) 候选行动收益严格定义为在真实候选证据池上执行一步取证行动 (H_t -> H_t + a) 后终端决策损失的实际下降量 Delta_L = L(H_t) - L(H_t + a)。"
            ),
        },
    }


def run_exp2_v2_non_circular(output_dir: str = "outputs") -> Dict[str, Any]:
    """
    修复后的 32 组受控合成仿真实验 (Exp2 v2)：
      - 引入完全独立于求解器、权重函数和规则系数的外部真值原子状态向量 z_gt in {0, 1}^8：
        * 旧图新用 (out_of_context): 图像画面本身为真 (z_0^+=1, z_0^-=0)，但时间、空间与整体事件归属均为假 (z_{1,2,3}^+=0, z_{1,2,3}^-=1)
        * 一致反驳 (unanimous_refute): 4个原子声明均为假 (z_{0..3}^+=0, z_{0..3}^-=1)
      - 分别独立报告：
        1. 相对外部真值 z_gt 的状态恢复误差 ||z*(E, P) - z_gt||_2 (检验各去重策略与软逻辑规则对恢复真实事实状态的真实贡献，无循环性)
        2. 相同规则参数 alpha 下的纯来源估计误差漂移 ||z*(E, P_hat) - z*(E, P_true)||_2 (检验不同来源折扣方法对 P 噪声的稳健性)
    """
    import itertools
    from src.core_models import (
        EvidenceSnippet,
        compute_discounted_weights,
        solve_convex_qp_inference,
        theoretical_stability_bound,
        compute_source_influence_feature,
    )
    from src.exp2_synthetic_32_simulation import build_cross_modal_rules, generate_single_controlled_case, paired_case_rng

    rng = np.random.default_rng(20261107)
    factor_grid = list(
        itertools.product(
            [2, 5],
            [False, True],
            ["unanimous_refute", "out_of_context"],
            [False, True],
            [0.06, 0.18],
        )
    )
    cases_v2 = []
    methods = [
        "independent",
        "domain_dedup",
        "max_quality_single",
        "semantic_dedup",
        "prob_discount",
        "oracle_source",
    ]

    for idx, (copy_k, same_dom, c_mode, u_rules, n_sigma) in enumerate(factor_grid, start=1):
        c_res = generate_single_controlled_case(
            case_idx=idx,
            copy_k=copy_k,
            same_domain_novel_fact=same_dom,
            conflict_mode=c_mode,
            use_rules=u_rules,
            noise_sigma=n_sigma,
            rng=paired_case_rng(copy_k, same_dom, c_mode),
        )
        # 外部独立真值状态 z_gt (完全不依赖 QP 求解器或 alpha 参数)
        if c_mode == "out_of_context":
            z_gt = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
        else:
            z_gt = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=float)

        for m_name in methods:
            z_star = np.array(c_res["methods"][m_name]["z_star"], dtype=float)
            err_vs_external_gt = float(np.linalg.norm(z_star - z_gt, ord=2))
            c_res["methods"][m_name]["error_vs_external_gt_l2"] = round(err_vs_external_gt, 5)
        c_res["external_z_gt"] = z_gt.tolist()
        cases_v2.append(c_res)

    summary_v2: Dict[str, Any] = {}
    for m_name in methods:
        ext_errs = [c["methods"][m_name]["error_vs_external_gt_l2"] for c in cases_v2]
        src_drifts = [c["methods"][m_name]["drift_vs_true_source_l2"] for c in cases_v2]
        margins = [c["methods"][m_name]["correct_refute_margin"] for c in cases_v2]
        same_dom_ret = [
            c["methods"][m_name]["novel_fact_retention_rate"]
            for c in cases_v2 if c["factors"]["same_domain_novel_fact"]
        ]
        summary_v2[m_name] = {
            "mean_error_vs_external_gt_l2": round(float(np.mean(ext_errs)), 4),
            "std_error_vs_external_gt_l2": round(float(np.std(ext_errs)), 4),
            "mean_pure_source_error_drift_l2": round(float(np.mean(src_drifts)), 4),
            "mean_correct_refute_margin": round(float(np.mean(margins)), 4),
            "same_domain_novel_fact_retention_rate": round(float(np.mean(same_dom_ret)), 4),
        }

    # 非循环评估软逻辑规则：对比 use_rules=True vs False 相对独立外部真值 z_gt 的误差
    gt_err_rules_on = [
        c["methods"]["prob_discount"]["error_vs_external_gt_l2"]
        for c in cases_v2 if c["factors"]["use_rules"]
    ]
    gt_err_rules_off = [
        c["methods"]["prob_discount"]["error_vs_external_gt_l2"]
        for c in cases_v2 if not c["factors"]["use_rules"]
    ]

    rep_v2 = {
        "experiment": "Exp2_v2_NonCircular_External_GroundTruth_Synthetic_32",
        "evidence_protocol": "CONTROLLED_SYNTHETIC_MECHANISM_ONLY",
        "formal_multimodal_results": False,
        "num_controlled_configurations": len(cases_v2),
        "summary_by_method": summary_v2,
        "non_circular_rule_ablation_vs_external_gt": {
            "rules_disabled_mean_error_vs_external_gt": round(float(np.mean(gt_err_rules_off)), 4),
            "rules_enabled_mean_error_vs_external_gt": round(float(np.mean(gt_err_rules_on)), 4),
            "relative_external_gt_error_reduction_by_rules": round(
                float((np.mean(gt_err_rules_off) - np.mean(gt_err_rules_on)) / np.mean(gt_err_rules_off)),
                4,
            ),
            "theoretical_stability_bound_100pct_satisfied": all(
                c["stability_verification"]["bound_satisfied"] for c in cases_v2
            ),
        },
        "cases": cases_v2,
    }

    with open(os.path.join(output_dir, "exp2_synthetic_32_v2_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep_v2, f, ensure_ascii=False, indent=2)
    return rep_v2


def run_phase1_audit(data_dir="data/averimatec", output_dir="outputs/method_repair"):
    """当前内容隔离审计；历史的静态审计说明不再作为当前代码合规结论。"""
    from pathlib import Path
    from src.data_integrity import audit_data, provisional_split
    root = Path(__file__).resolve().parents[1]
    if (root/data_dir).resolve() != (root/"data/averimatec").resolve():
        raise ValueError("This audit is bound to the versioned AVerImaTeC workspace")
    report = audit_data(root)
    target = root/output_dir
    target.mkdir(parents=True, exist_ok=True)
    (target/"dataset_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (target/"provisional_split.json").write_text(json.dumps(provisional_split(report),ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    from run_all_preliminary_experiments import main
    raise SystemExit(main())
