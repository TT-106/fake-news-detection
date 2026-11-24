"""重新从数据内容核查事件隔离，不相信报告中的零泄漏布尔值。"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from src.exp1_averimatec_audit_and_split import normalize_url_for_matching, tokenize_entities
from src.research_contracts import ProtocolError


AI_EVENT_PROTOCOL = "EVENT_REVIEW_AB_AI_ADJUDICATION_V1"
AI_EVENT_APPROVAL_SHA = "fc9d5cf267b515a078a99a06b5a3dc8e02afe7cf8bc5e773d0c7c81c9deea6ec"


def _verified_json(root, ref):
    path = (root/ref["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ProtocolError("协议文件路径越界")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ProtocolError(f"协议材料摘要不同：{path.name}")
    return json.loads(raw)


def event_split_approved(split, root):
    if split.get("protocol_id") != AI_EVENT_PROTOCOL:
        return split.get("human_event_review_complete") is True
    ref = split.get("review_protocol", {})
    if (ref.get("sha256") != AI_EVENT_APPROVAL_SHA or split.get("event_review_complete") is not True
            or split.get("formal_split_applied") is not True or split.get("human_event_review_complete") is not False):
        return False
    approval = _verified_json(root, ref)
    candidate = _verified_json(root, approval["candidate"])
    decisions = _verified_json(root, approval["adjudication"])
    for original in approval["raw_returns"]:
        _verified_json(root, original)
    return (approval["protocol_id"] == AI_EVENT_PROTOCOL and decisions["reviewer_type"] == "ai"
            and all(split[k] == candidate[k] for k in ("split", "event_groups", "quarantined_train_claims",
                    "counts", "dataset_sha256", "source_receipts", "source_provisional_split", "unresolved_review_ids"))
            and set(approval["unresolved_disposition"].values()) <= set(split["quarantined_train_claims"]))


def audit_data(root: Path, *, include_final_eval=False) -> dict:
    locked = root/"data/annotations/experiment_split.json"
    if locked.is_file() and not include_final_eval:
        split = json.loads(locked.read_bytes())
        if split.get("protocol_id") == AI_EVENT_PROTOCOL:
            if not event_split_approved(split, root):
                raise ProtocolError("AI 事件裁决协议或固定划分验证失败")
            from src.learned_components import ExperimentSplit
            old = _verified_json(root, split["source_provisional_split"])
            ref = next(r for r in split["source_receipts"] if r["path"] == "outputs/method_repair/dataset_audit.json")
            prior = _verified_json(root, ref)
            train = (root/"data/averimatec/train.json").read_bytes()
            if (hashlib.sha256(train).hexdigest() != split["dataset_sha256"]["train.json"]
                    or prior["dataset_sha256"] != split["dataset_sha256"] or prior["event_groups"] != old["event_groups"]):
                raise ProtocolError("训练文件或封存前数据审计不匹配")
            obj = ExperimentSplit(**{k:frozenset(v) for k,v in split["split"].items()})
            obj.validate(old["event_groups"])
            obj.validate(split["event_groups"])
            if obj.final_eval != frozenset(f"val:{i}" for i in range(152)):
                raise ProtocolError("终评 ID 与封存清单不符")
            return {"protocol": AI_EVENT_PROTOCOL, "dataset_sha256": prior["dataset_sha256"],
                    "event_groups": split["event_groups"], "isolation_status": "PASS_FOR_OBSERVED_KEYS",
                    "val_verification": "PRESEAL_RECEIPT_AND_FIXED_IDS", "val_file_opened": False}
    train_path, val_path = root/"data/averimatec/train.json", root/"data/averimatec/val.json"
    train = json.loads(train_path.read_text(encoding="utf-8"))
    val = json.loads(val_path.read_text(encoding="utf-8"))
    samples = train+val
    ids = [f"train:{i}" for i in range(len(train))]+[f"val:{i}" for i in range(len(val))]
    index_path = root/"outputs/image_zip_crc_index.json"
    crc = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    parent = list(range(len(samples)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)
    owner, missing_crc, sample_keys, tokens = {}, set(), [], []
    for i, s in enumerate(samples):
        keys = set()
        text = " ".join((s.get("claim_text") or "").casefold().split())
        if text:
            keys.add(("claim", text))
        for u in (s.get("article"), (s.get("metadata") or {}).get("original_claim_url")):
            if u:
                keys.add(("url", normalize_url_for_matching(u)))
        images = list(s.get("claim_images") or [])
        for q in s.get("questions") or []:
            images.extend(q.get("input_images") or [])
            for a in q.get("answers") or []:
                u = a.get("source_url") or ""
                if u.startswith(("http://", "https://")):
                    keys.add(("url", normalize_url_for_matching(u)))
        for name in images:
            base = name.replace("\\", "/").rsplit("/", 1)[-1]
            meta = crc.get(base)
            if meta is None:
                missing_crc.add(base)
                keys.add(("image_filename_only", base))
            else:
                keys.add(("image_crc32_size", str(meta["crc32"]), str(meta["uncomp_size"])))
        sample_keys.append(keys)
        for k in keys:
            if k in owner:
                union(i, owner[k])
            else:
                owner[k] = i
        tokens.append(tokenize_entities(s.get("claim_text") or ""))
    for i, a in enumerate(tokens):
        if len(a) < 3:
            continue
        for j in range(i+1, len(tokens)):
            b = tokens[j]
            common = len(a & b)
            if common >= 4 and common/max(1, len(a | b)) >= .55:
                union(i, j)
    groups = {cid: f"event:{find(i)}" for i, cid in enumerate(ids)}
    train_groups, val_groups = {groups[c] for c in ids[:len(train)]}, {groups[c] for c in ids[len(train):]}
    crossings = train_groups & val_groups
    split_path = root/"outputs/averimatec_event_isolated_splits_v2.json"
    role_ids, issues = {}, []
    if split_path.is_file():
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        assigned = {}
        for role, info in split_data["splits"].items():
            role_ids[role] = []
            for i in info["indices"]:
                cid = f"train:{i}"
                if i < 0 or i >= len(train) or cid in assigned:
                    raise ProtocolError("划分包含越界或重复训练样本索引")
                assigned[cid] = role
                role_ids[role].append(cid)
        if set(assigned) != set(ids[:len(train)]):
            issues.append("部分训练样本未分配用途")
        group_roles = defaultdict(set)
        for cid, role in assigned.items():
            group_roles[groups[cid]].add(role)
        for g, roles in group_roles.items():
            if len(roles) > 1:
                issues.append({"event_group": g, "roles": sorted(roles)})
    else:
        issues.append("缺少事件组用途划分")
    return {"protocol": "DATA_CONTENT_AUDIT_ONLY", "dataset_sha256": {
                "train.json": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "val.json": hashlib.sha256(val_path.read_bytes()).hexdigest()},
            "train": {"count": len(train), "labels": dict(Counter(s["label"] for s in train))},
            "val": {"count": len(val), "labels": dict(Counter(s["label"] for s in val))},
            "train_val_shared_event_groups": sorted(crossings), "internal_split_issues": issues,
            "missing_image_crc_entries": sorted(missing_crc),
            "image_identity_check": "ZIP_CRC32_AND_UNCOMPRESSED_SIZE_METADATA_ONLY; not SHA256 or decoded image validation",
            "webpage_retrieval_check": "NOT_PERFORMED; annotated URLs are used only to detect grouping overlap",
            "isolation_status": "PASS_FOR_OBSERVED_KEYS" if not crossings and not issues and not missing_crc else "BLOCKED_OR_INCOMPLETE",
            "event_groups": groups, "role_claim_ids": role_ids,
            "grouping_caveat": "CRC32/URL/text heuristics can miss semantic event overlap; manual event review remains required"}


def provisional_split(audit: dict, seed: int = 42) -> dict:
    """保留官方评测集，隔离与其关联的训练簇；候选划分不冒充人工事件审阅。"""
    import numpy as np
    groups = audit["event_groups"]
    official = sorted(cid for cid in groups if cid.startswith("val:"))
    final_groups = {groups[c] for c in official}
    quarantine = sorted(cid for cid in groups if cid.startswith("train:") and groups[cid] in final_groups)
    candidates = defaultdict(list)
    for cid, gid in groups.items():
        if cid.startswith("train:") and cid not in quarantine:
            candidates[gid].append(cid)
    previous = {cid: role for role, ids in audit["role_claim_ids"].items() for cid in ids}
    targets = {"dev_calibration": 60, "pilot_probe": 20, "action_train": 180, "mechanism_val": 60}
    assigned = {name: [] for name in targets}
    remaining = []
    # 先保留不冲突、未污染的原用途组，再用储备组补足；不按实验成绩挑样本。
    for gid, ids in sorted(candidates.items()):
        roles = {previous.get(cid, "reserve_pool") for cid in ids}
        role = next(iter(roles)) if len(roles) == 1 else "reserve_pool"
        if role in targets and len(assigned[role])+len(ids) <= targets[role]:
            assigned[role].extend(ids)
        else:
            remaining.append((gid, ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    for role, target in targets.items():
        while len(assigned[role]) < target:
            size = target-len(assigned[role])
            match = next((i for i, (_, ids) in enumerate(remaining) if len(ids) <= size), None)
            if match is None:
                raise ProtocolError("无法在保持事件组完整的同时满足样本数；须调整预注册划分数量")
            _, ids = remaining.pop(match)
            assigned[role].extend(ids)
    dev_groups = defaultdict(list)
    for cid in assigned["dev_calibration"]:
        dev_groups[groups[cid]].append(cid)
    dev = list(sorted(dev_groups.items()))
    rng.shuffle(dev)
    calibration, fit = [], []
    for _, ids in dev:
        (calibration if len(calibration)+len(ids) <= 15 else fit).extend(ids)
    split = {"module_fit": sorted(fit), "temperature_calibration": sorted(calibration),
             "pilot": sorted(assigned["pilot_probe"]), "action_train": sorted(assigned["action_train"]),
             "mechanism_validation": sorted(assigned["mechanism_val"]), "final_eval": official,
             "reserve": sorted(cid for _, ids in remaining for cid in ids)}
    from src.learned_components import ExperimentSplit
    ExperimentSplit(**{k: frozenset(v) for k, v in split.items()}).validate(groups)
    new_roles = {cid: role for role, ids in split.items() for cid in ids}
    changes = [{"claim_id": cid, "previous_role": previous.get(cid), "new_role": new_roles.get(cid, "QUARANTINED")}
               for cid in sorted(previous)
               if new_roles.get(cid, "QUARANTINED") != {"dev_calibration": "module_fit", "pilot_probe": "pilot",
                                                       "mechanism_val": "mechanism_validation", "reserve_pool": "reserve"}.get(previous[cid], previous[cid])]
    return {"status": "PROVISIONAL_REQUIRES_MANUAL_EVENT_REVIEW", "seed": seed,
            "dataset_sha256": audit["dataset_sha256"], "event_groups": groups, "split": split,
            "quarantined_train_claims": quarantine, "role_changes": changes,
            "counts": {**{role: len(ids) for role, ids in split.items()}, "quarantined": len(quarantine)},
            "official_eval_kept_unchanged": True, "human_event_review_complete": False,
            "note": "隔离了与官方评测集共享关联组的训练样本；原始数据不变，储备集人数相应减少。"}
