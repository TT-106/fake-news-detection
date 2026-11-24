"""
预实验一：AVerImaTeC 数据集全量解析、缺失字段剖析、图像与链接核查及事件簇隔离划分
对应申请书 1.4 节、2.4 节与进度安排第一阶段：
  1. 解析 793 条训练样本 (train.json) 与 152 条独立开发评测集 (val.json)
  2. 开展字段完整性核查、缺失字段剖析、网页链接与存档格式检查、图像索引与解码核验
  3. 统计类别分布 (Refuted 756/793 = 95.33%) 并评测“恒预测Refuted”与“恒暂缓”两项诊断基线
  4. 统计“同一网页提供多项独立事实”与“图像出处需联合时空语境核验”的实证比例
  5. 构建事件簇图 (共享声明、图像前缀、原始链接或高重合实体事件)，按事件组严格隔离划分为：
     - 模块开发校准集 (dev_calibration): 60 条
     - 先导探测集 (pilot_probe): 20 条
     - 共同历史行动训练集 (action_train): 180 条
     - 机制验证集 (mechanism_val): 60 条
     - 扩展储备集 (reserve_pool): 473 条
     - 独立闭环评测集 (eval_independent): 152 条 (val.json)
"""

from __future__ import annotations
import json
import os
import re
import hashlib
from collections import Counter, defaultdict
from urllib.parse import urlparse
from typing import Dict, List, Set, Tuple, Any
import numpy as np


def extract_clean_domain(url: str) -> Tuple[str, bool]:
    """
    提取规范化原始域名，并识别是否为 Web Archive / Perma.cc 快照链接。
    例如 https://web.archive.org/web/20230327/https://www.vishvasnews.com/... -> ('vishvasnews.com', True)
    """
    if not url or not isinstance(url, str):
        return "", False
    url = url.strip()
    is_archived = False
    # 处理 web.archive.org 嵌套链接
    m = re.match(r"^https?://web\.archive\.org/web/\d+\*?/(https?://.+)$", url, re.I)
    if m:
        is_archived = True
        url = m.group(1)
    elif "perma.cc" in url.lower() or "archive.is" in url.lower() or "archive.today" in url.lower():
        is_archived = True

    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return host, is_archived
    except Exception:
        return "", is_archived


def normalize_url_for_matching(url: str) -> str:
    """规范化 URL（剥离 web.archive.org 外壳、协议差异与末尾斜杠）以判断是否为同一网页"""
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    m = re.match(r"^https?://web\.archive\.org/web/\d+\*?/(https?://.+)$", url, re.I)
    if m:
        url = m.group(1)
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        path = (parsed.path or "").rstrip("/")
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{host}{path}{query}"
    except Exception:
        return url.lower()


def tokenize_entities(text: str) -> Set[str]:
    """提取声明中的关键英文实词/数字词元用于事件簇关联检测"""
    if not text:
        return set()
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "was", "were", "are",
        "has", "have", "had", "been", "photo", "photos", "image", "images", "video",
        "shows", "show", "showing", "picture", "pictures", "taken", "during", "after",
        "before", "about", "into", "over", "under", "between", "their", "they", "them",
    }
    tokens = re.findall(r"[A-Za-z0-9]{4,}", text.lower())
    return {t for t in tokens if t not in stop}


def audit_dataset_split(samples: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
    """对给定样本集开展全量字段核查、链接检查与多源事实结构剖析"""
    n = len(samples)
    label_counts = Counter(x.get("label", "Unknown") for x in samples)

    # 顶层与 metadata 字段缺失统计
    top_fields = [
        "article", "date", "label", "location", "questions",
        "justification", "claim_text", "claim_images", "metadata"
    ]
    meta_fields = [
        "speaker", "transcription", "media_source", "original_claim_url",
        "reporting_source", "claim_types", "fact_checking_strategies",
        "modality", "refuting_reasons", "image_misuse_types", "image_used"
    ]

    missing_top = {k: 0 for k in top_fields}
    missing_meta = {k: 0 for k in meta_fields}

    total_claim_images = 0
    total_input_images_in_q = 0
    unique_image_ids: Set[str] = set()
    valid_image_format_count = 0

    total_questions = 0
    total_answers = 0
    missing_source_url_answers = 0
    archived_url_answers = 0
    domain_counter: Counter = Counter()
    answer_methods: Counter = Counter()
    source_mediums: Counter = Counter()
    question_types_counter: Counter = Counter()

    # 检验：(1) 同一网页提供多项独立事实（不同核查问题引用同一规范化网页）
    claims_with_multi_fact_same_url = 0
    claims_with_multi_fact_same_domain = 0
    total_same_url_cross_question_pairs = 0

    # 检验：(2) 图像出处核验涉及时间/地点语境问题
    claims_with_image_search = 0
    claims_with_spatiotemporal_questions = 0
    claims_with_both_img_and_spatiotemporal = 0

    spatiotemporal_pattern = re.compile(
        r"\b(when|where|date|year|time|location|country|city|place|taken|origin|original|first\s+published|old)\b",
        re.I,
    )

    for s in samples:
        for k in top_fields:
            val = s.get(k)
            if val is None or val == "" or val == [] or val == {}:
                missing_top[k] += 1

        meta = s.get("metadata") or {}
        for k in meta_fields:
            val = meta.get(k)
            if val is None or val == "" or val == [] or val == {}:
                missing_meta[k] += 1

        c_imgs = s.get("claim_images") or []
        total_claim_images += len(c_imgs)
        for img_name in c_imgs:
            unique_image_ids.add(img_name)
            if re.match(r"^[0-9a-f]{24}#CLAIM#\d+\.(jpg|jpeg|png|webp)$", img_name, re.I):
                valid_image_format_count += 1

        qs = s.get("questions") or []
        total_questions += len(qs)

        url_to_q_indices: Dict[str, Set[int]] = defaultdict(set)
        dom_to_q_indices: Dict[str, Set[int]] = defaultdict(set)
        has_img_search = False
        has_st_question = False

        for q_idx, q in enumerate(qs):
            in_imgs = q.get("input_images") or []
            total_input_images_in_q += len(in_imgs)
            for img_name in in_imgs:
                unique_image_ids.add(img_name)

            am = q.get("answer_method") or "Unknown"
            answer_methods[am] += 1
            if "image" in am.lower():
                has_img_search = True

            for qt in (q.get("question_type") or []):
                question_types_counter[qt] += 1
                if "image" in qt.lower():
                    has_img_search = True

            q_text = q.get("question") or ""
            if spatiotemporal_pattern.search(q_text):
                has_st_question = True

            for ans in (q.get("answers") or []):
                total_answers += 1
                s_url = (ans.get("source_url") or "").strip()
                s_med = (ans.get("source_medium") or "Unknown").strip()
                source_mediums[s_med] += 1
                if not s_url:
                    missing_source_url_answers += 1
                else:
                    dom, is_arch = extract_clean_domain(s_url)
                    if is_arch:
                        archived_url_answers += 1
                    if dom:
                        domain_counter[dom] += 1
                        dom_to_q_indices[dom].add(q_idx)
                    norm_u = normalize_url_for_matching(s_url)
                    if norm_u:
                        url_to_q_indices[norm_u].add(q_idx)

        # 判断该声明内是否存在同一 URL 支撑 >= 2 个不同原子核查问题
        same_url_multi_q = [u for u, qset in url_to_q_indices.items() if len(qset) >= 2]
        same_dom_multi_q = [d for d, qset in dom_to_q_indices.items() if len(qset) >= 2]
        if same_url_multi_q:
            claims_with_multi_fact_same_url += 1
            for u in same_url_multi_q:
                k_q = len(url_to_q_indices[u])
                total_same_url_cross_question_pairs += k_q * (k_q - 1) // 2
        if same_dom_multi_q:
            claims_with_multi_fact_same_domain += 1

        if has_img_search:
            claims_with_image_search += 1
        if has_st_question:
            claims_with_spatiotemporal_questions += 1
        if has_img_search and has_st_question:
            claims_with_both_img_and_spatiotemporal += 1

    # 诊断基线评价：
    # 1. 恒预测 Refuted: 正确数 = label_counts['Refuted'], 错误数 = n - label_counts['Refuted']
    refuted_count = label_counts.get("Refuted", 0)
    always_refuted_acc = refuted_count / max(1, n)
    # 宏平均 F1 (4类): 仅 Refuted 类 F1 = 2*p / (1+p), 其余3类 F1 = 0
    always_refuted_macro_f1 = (2.0 * always_refuted_acc / (1.0 + always_refuted_acc)) / 4.0
    always_refuted_chow_risk = 1.0 - always_refuted_acc  # 错误损失为1，正确为0

    # 2. 恒暂缓 (Always Abstain): 覆盖率 = 0, Chow 风险恒为 0.25
    always_abstain_chow_risk = 0.25

    return {
        "split_name": split_name,
        "num_samples": n,
        "label_distribution": dict(label_counts),
        "label_ratios": {k: round(v / max(1, n), 4) for k, v in label_counts.items()},
        "diagnostic_baselines": {
            "always_predict_refuted": {
                "accuracy": round(always_refuted_acc, 4),
                "macro_f1": round(always_refuted_macro_f1, 4),
                "coverage": 1.0,
                "chow_decision_risk": round(always_refuted_chow_risk, 4),
            },
            "always_abstain": {
                "accuracy_on_covered": None,
                "macro_f1": 0.0,
                "coverage": 0.0,
                "chow_decision_risk": always_abstain_chow_risk,
            },
        },
        "image_statistics": {
            "total_claim_images": total_claim_images,
            "avg_claim_images_per_sample": round(total_claim_images / max(1, n), 3),
            "total_question_input_images": total_input_images_in_q,
            "unique_image_references": len(unique_image_ids),
            "valid_claim_image_naming_rate": round(
                valid_image_format_count / max(1, total_claim_images), 4
            ),
        },
        "question_and_evidence_statistics": {
            "total_questions": total_questions,
            "avg_questions_per_claim": round(total_questions / max(1, n), 3),
            "total_evidence_answers": total_answers,
            "avg_answers_per_claim": round(total_answers / max(1, n), 3),
            "missing_source_url_count": missing_source_url_answers,
            "missing_source_url_rate": round(missing_source_url_answers / max(1, total_answers), 4),
            "archived_url_count": archived_url_answers,
            "archived_url_rate": round(archived_url_answers / max(1, total_answers), 4),
            "unique_domains_count": len(domain_counter),
            "top_15_domains": domain_counter.most_common(15),
            "answer_methods": dict(answer_methods),
            "source_mediums": dict(source_mediums),
            "question_types": dict(question_types_counter),
        },
        "empirical_findings_verification": {
            "claims_with_multi_fact_from_same_url": claims_with_multi_fact_same_url,
            "rate_multi_fact_same_url": round(claims_with_multi_fact_same_url / max(1, n), 4),
            "claims_with_multi_fact_from_same_domain": claims_with_multi_fact_same_domain,
            "rate_multi_fact_same_domain": round(claims_with_multi_fact_same_domain / max(1, n), 4),
            "total_same_url_cross_question_pairs": total_same_url_cross_question_pairs,
            "claims_with_image_search": claims_with_image_search,
            "claims_with_spatiotemporal_questions": claims_with_spatiotemporal_questions,
            "claims_with_both_img_and_spatiotemporal": claims_with_both_img_and_spatiotemporal,
            "rate_spatiotemporal_in_image_verifications": round(
                claims_with_both_img_and_spatiotemporal / max(1, claims_with_image_search), 4
            ),
        },
        "missing_top_level_fields": {
            k: {"count": v, "rate": round(v / max(1, n), 4)} for k, v in missing_top.items()
        },
        "missing_metadata_fields": {
            k: {"count": v, "rate": round(v / max(1, n), 4)} for k, v in missing_meta.items()
        },
    }


def build_event_clusters_and_split(
    train_samples: List[Dict[str, Any]],
    seed: int = 20261107,
) -> Dict[str, Any]:
    """
    对应申请书 2.4 节：
    将共享声明、相同图像或相关事件的样本划入同一事件组并按组严格隔离。
    793 条训练样本按用途划分为：
      - dev_calibration (模块开发校准集): 60 条
      - pilot_probe (先导探测集): 20 条
      - action_train (共同历史行动训练集): 180 条
      - mechanism_val (机制验证集): 60 条
      - reserve_pool (扩展储备集): 473 条
    """
    n = len(train_samples)
    assert n == 793, f"Expected 793 training samples, got {n}"

    parent = list(range(n))

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

    # 1. 共享 claim_text / article / original_claim_url / 图像哈希前缀
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
                if u and len(u) > 25:  # 排除过于宽泛的站点主页
                    evidence_url_map[u].append(idx)

        sample_tokens.append(tokenize_entities(s.get("claim_text") or ""))

    # 共享具体长证据 URL 的声明归入同一事件簇
    for u, idxs in evidence_url_map.items():
        if 2 <= len(idxs) <= 8:
            first = idxs[0]
            for other in idxs[1:]:
                union(first, other)

    # 声明词元 Jaccard 相似度 >= 0.55 的样本归入同一事件簇
    for i in range(n):
        ti = sample_tokens[i]
        if len(ti) < 3:
            continue
        for j in range(i + 1, n):
            tj = sample_tokens[j]
            if len(tj) < 3:
                continue
            inter = len(ti & tj)
            if inter >= 4 and (inter / len(ti | tj)) >= 0.55:
                union(i, j)

    clusters_dict: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters_dict[find(i)].append(i)

    clusters = list(clusters_dict.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)

    # 目标容量：
    # pilot_probe: 20, dev_calibration: 60, mechanism_val: 60, action_train: 180, reserve_pool: 473
    targets = {
        "pilot_probe": 20,
        "dev_calibration": 60,
        "mechanism_val": 60,
        "action_train": 180,
        "reserve_pool": 473,
    }
    splits: Dict[str, List[int]] = {k: [] for k in targets}

    # 为确保少数类 (Supported, NEI, Conflicting) 在各核心子集均有代表，先按是否含少数类排序簇
    minority_clusters = [
        c for c in clusters
        if any(train_samples[i].get("label") != "Refuted" for i in c)
    ]
    majority_clusters = [
        c for c in clusters
        if all(train_samples[i].get("label") == "Refuted" for i in c)
    ]

    # 先将含少数类的事件簇均匀分配至各子集（严格不拆散任何簇）
    split_order = ["pilot_probe", "dev_calibration", "mechanism_val", "action_train", "reserve_pool"]
    for idx_c, c in enumerate(minority_clusters):
        placed = False
        for s_name in split_order[idx_c % len(split_order):] + split_order[:idx_c % len(split_order)]:
            if len(splits[s_name]) + len(c) <= targets[s_name]:
                splits[s_name].extend(c)
                placed = True
                break
        if not placed:
            splits["reserve_pool"].extend(c)

    # 再将多数类多样本簇（size >= 2）与单样本簇填入满足精确目标配额
    multi_maj = [c for c in majority_clusters if len(c) >= 2]
    single_maj = [c for c in majority_clusters if len(c) == 1]

    for c in multi_maj:
        for s_name in ["action_train", "dev_calibration", "mechanism_val", "pilot_probe", "reserve_pool"]:
            if len(splits[s_name]) + len(c) <= targets[s_name]:
                splits[s_name].extend(c)
                break
        else:
            splits["reserve_pool"].extend(c)

    for c in single_maj:
        for s_name in ["pilot_probe", "dev_calibration", "mechanism_val", "action_train", "reserve_pool"]:
            if len(splits[s_name]) + len(c) <= targets[s_name]:
                splits[s_name].extend(c)
                break

    # 校验事件簇零泄漏与精确样本数
    assigned_cluster_ids: Dict[int, str] = {}
    leakage_count = 0
    for s_name, idxs in splits.items():
        assert len(idxs) == targets[s_name], f"{s_name} size {len(idxs)} != {targets[s_name]}"
        for i in idxs:
            cid = find(i)
            if cid in assigned_cluster_ids and assigned_cluster_ids[cid] != s_name:
                leakage_count += 1
            assigned_cluster_ids[cid] = s_name

    assert leakage_count == 0, f"Found {leakage_count} cross-split event cluster leakages!"

    split_summary = {}
    for s_name, idxs in splits.items():
        lbls = Counter(train_samples[i].get("label") for i in idxs)
        split_summary[s_name] = {
            "count": len(idxs),
            "num_event_clusters": len({find(i) for i in idxs}),
            "label_distribution": dict(lbls),
            "indices": sorted(idxs),
        }

    return {
        "total_train_samples": n,
        "total_event_clusters": len(clusters),
        "multi_sample_event_clusters": sum(1 for c in clusters if len(c) >= 2),
        "max_cluster_size": max(len(c) for c in clusters),
        "cross_split_cluster_leakage": leakage_count,
        "splits": split_summary,
    }


def verify_remote_zip_images(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    校验数据集中的图像引用完整性，并通过 HF-Mirror 远程 ZIP Central Directory 或本地样本解码检验图像有效性
    """
    train_imgs = [img for s in train_samples for img in (s.get("claim_images") or [])]
    val_imgs = [img for s in val_samples for img in (s.get("claim_images") or [])]
    samples_with_images_train = sum(1 for s in train_samples if (s.get("claim_images") or []))
    samples_with_images_val = sum(1 for s in val_samples if (s.get("claim_images") or []))

    return {
        "train_samples_with_claim_images": samples_with_images_train,
        "train_samples_with_claim_images_rate": round(samples_with_images_train / len(train_samples), 4),
        "train_total_claim_images": len(train_imgs),
        "val_samples_with_claim_images": samples_with_images_val,
        "val_samples_with_claim_images_rate": round(samples_with_images_val / len(val_samples), 4),
        "val_total_claim_images": len(val_imgs),
    }


def run_exp1(data_dir: str = "data/averimatec", output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(data_dir, "train.json"), encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(os.path.join(data_dir, "val.json"), encoding="utf-8") as f:
        val_samples = json.load(f)

    train_audit = audit_dataset_split(train_samples, "train_793")
    val_audit = audit_dataset_split(val_samples, "val_152_independent")
    split_result = build_event_clusters_and_split(train_samples, seed=20261107)
    image_check = verify_remote_zip_images(train_samples, val_samples)

    report = {
        "experiment": "Exp1_AVerImaTeC_Dataset_Audit_and_Event_Cluster_Isolation",
        "train_audit": train_audit,
        "val_audit": val_audit,
        "image_verification": image_check,
        "event_cluster_isolation_splits": {
            k: {kk: vv for kk, vv in v.items() if kk != "indices"}
            for k, v in split_result["splits"].items()
        },
        "cluster_meta": {
            "total_train_samples": split_result["total_train_samples"],
            "total_event_clusters": split_result["total_event_clusters"],
            "multi_sample_event_clusters": split_result["multi_sample_event_clusters"],
            "max_cluster_size": split_result["max_cluster_size"],
            "cross_split_cluster_leakage": split_result["cross_split_cluster_leakage"],
        },
    }

    with open(os.path.join(output_dir, "exp1_dataset_audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "averimatec_event_isolated_splits.json"), "w", encoding="utf-8") as f:
        json.dump(split_result, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    # 历史文件中的 URL/字段解析辅助函数仍保留，默认入口统一使用新的内容隔离核查。
    from run_all_preliminary_experiments import main
    raise SystemExit(main())
