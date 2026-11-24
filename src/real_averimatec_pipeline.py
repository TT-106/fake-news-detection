"""正式实验入口及显式标记的金标准 QA 诊断适配器。

旧的合成回声室、人工收益、替代 VLM 分类器路径已隔离至修复前备份。
默认只核查准备状态；缺少真实材料不能生成 E1—E4 成绩。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.core_models import EvidenceSnippet
from src.data_integrity import audit_data, event_split_approved
from src.exp1_averimatec_audit_and_split import extract_clean_domain
from src.model_inference import get_nli_model
from src.research_contracts import ProtocolError, fingerprint


ROOT_DIR = Path(__file__).resolve().parents[1]


def strip_ground_truth_fields(sample: dict) -> dict:
    """正式推理输入白名单。包括 QA 和金标准 URL 在内的标注均不进入。"""
    return {"claim_text": sample.get("claim_text") or "", "date": sample.get("date") or "",
            "location": sample.get("location") or "", "claim_images": list(sample.get("claim_images") or [])}


def build_claim_evidence_environment(sample: dict, sample_id: int,
                                     image_meta_map: dict | None = None,
                                     ablation_mode: str = "full") -> dict:
    """兼容旧调用的 QA 诊断，不是正式开放网页实验，不用于训练或评价。

    只解析已有答案，不补齐、不生成网页、不把时空槽位当事实、不读取真值。
    NLI 只给出文本诊断。质量模型未训练，因此质量置零表示不具备正式赋权资格，
    并明确返回 formal_evaluation_eligible=False；引文未取得快照即不可回放。
    """
    if ablation_mode != "full":
        raise ProtocolError("QA 文本诊断不支持冒充多模态消融；正式消融须使用固定 VLM 和相同真实证据包")
    obs = strip_ground_truth_fields(sample)
    records, snippets, seen = [], [], set()
    for q in sample.get("questions") or []:
        question = q.get("question") or ""
        for a in q.get("answers") or []:
            text = ((a.get("answer_text") or "")+" "+(a.get("boolean_explanation") or "")).strip()
            url = a.get("source_url") or ""
            key = (url, text, question)
            if not text or key in seen:
                continue
            seen.add(key)
            if len(records) == 8:
                continue
            domain, archived = extract_clean_domain(url)
            eid = len(records)
            fact = fingerprint({"question": question.casefold(), "answer": text.casefold()})
            records.append({"rec_id": eid, "question": question, "answer_text": a.get("answer_text") or "",
                            "boolean_explanation": a.get("boolean_explanation") or "", "url": url,
                            "domain": domain, "is_archived": archived, "fact_id": fact,
                            "image_crc_match": None, "source_protocol": "ANNOTATION_ASSISTED_GOLD_QA"})
            entail, contradict, _ = get_nli_model().stance(text, obs["claim_text"])
            snippets.append(EvidenceSnippet(eid, domain, url, fact, np.zeros(1),
                                            np.array([entail]), np.array([contradict]),
                                            timestamp_verified=False, url_replayable=False))
    return {"sample_id": sample_id, "obs_sample": obs, "raw_records": records, "snippets": snippets,
            "evidence_protocol": "ANNOTATION_ASSISTED_GOLD_QA", "formal_evaluation_eligible": False,
            "quality_model_status": "UNTRAINED_DIAGNOSTIC_ZERO_WEIGHT",
            "fact_identity_status": "QUESTION_ANSWER_IDENTITY_ONLY_NOT_CANONICAL_VLM_FACTS",
            "pair_labels": [-1]*(len(snippets)*(len(snippets)-1)//2),
            "P_true": None, "citation_replay_status": "NO_ARCHIVED_PASSAGE_PROOF"}


def check_readiness(config_path: Path | None = None) -> dict:
    config_path = config_path or ROOT_DIR/"config/experiment.json"
    reasons = []
    config = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        reasons.append({"component": "experiment_config", "status": "MISSING", "path": str(config_path)})
    model_path = ROOT_DIR/config.get("model_dir", "models/Qwen2.5-VL-7B-Instruct")
    if not (model_path/"config.json").is_file() or not list(model_path.glob("*.safetensors")):
        reasons.append({"component": "frozen_vision_language_model", "status": "MISSING_LOCAL_WEIGHTS",
                        "path": str(model_path), "substitution_allowed": False, "download_attempted": False})
    required = {"source_annotations": "data/annotations/source.jsonl",
                "quality_annotations": "data/annotations/quality.jsonl",
                "sufficiency_annotations": "data/annotations/sufficiency.jsonl",
                "experiment_split": "data/annotations/experiment_split.json",
                "archived_tools": "data/open_web/tool_manifest.json"}
    for key, default in required.items():
        p = ROOT_DIR/config.get(key, default)
        if not p.is_file() or p.stat().st_size == 0:
            reasons.append({"component": key, "status": "MISSING_REAL_DATA", "path": str(p)})
    image_dir = ROOT_DIR/config.get("image_dir", "data/averimatec/images")
    if not image_dir.is_dir() or not any(image_dir.iterdir()):
        reasons.append({"component": "original_claim_images", "status": "MISSING_LOCAL_IMAGES", "path": str(image_dir)})
    try:
        audit = audit_data(ROOT_DIR)
    except FileNotFoundError as exc:
        audit = None
        reasons.append({"component": "dataset_audit", "status": "MISSING_AUDIT_INPUT", "path": exc.filename})
    split_file = ROOT_DIR/config.get("experiment_split", "data/annotations/experiment_split.json")
    if audit is not None and split_file.is_file():
        from src.learned_components import ExperimentSplit
        try:
            split_data = json.loads(split_file.read_text(encoding="utf-8"))
            plan = ExperimentSplit(**{k: frozenset(v) for k,v in split_data["split"].items()})
            plan.validate(split_data["event_groups"])
            if not event_split_approved(split_data, ROOT_DIR):
                reasons.append({"component": "event_split", "status": "AWAITING_MANUAL_EVENT_REVIEW"})
            if split_data.get("dataset_sha256") != audit["dataset_sha256"]:
                reasons.append({"component": "event_split", "status": "DATASET_VERSION_MISMATCH"})
            seen = {}
            for role, cids in vars(plan).items():
                for cid in cids:
                    gid = audit["event_groups"][cid]
                    if gid in seen and seen[gid] != role:
                        raise ProtocolError("关联组跨用途")
                    seen[gid] = role
        except (ValueError, KeyError, TypeError) as exc:
            reasons.append({"component": "event_split", "status": "INVALID_OR_LEAKED_SPLIT", "reason": str(exc)})
    elif audit is not None and audit["isolation_status"] != "PASS_FOR_OBSERVED_KEYS":
        reasons.append({"component": "event_split", "status": audit["isolation_status"]})
    status = "BLOCKED_MISSING_PREREQUISITES" if reasons else "FILES_PRESENT_REQUIRES_VALIDATED_RUN"
    return {"status": status, "protocol_version": "APPLICATION_2_2_FAITHFUL_V1",
            "formal_experiments_executed": False, "formal_scores": None, "blockers": reasons,
            "dataset_audit": audit, "config_path": str(config_path),
            "note": "文件存在不等于标注、模型或工具轨迹已经验证；实际运行仍会逐条校验。"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="按申请书方法执行实验；默认仅核查准备状态，不下载模型")
    parser.add_argument("--config", type=Path, default=ROOT_DIR/"config/experiment.json")
    parser.add_argument("--run", action="store_true", help="材料齐备后运行真实归档轨迹实验")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR/"outputs/preflight")
    args = parser.parse_args(argv)
    report = check_readiness(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "formal_scores": None, "blockers": report["blockers"]},
                     ensure_ascii=False, indent=2))
    if report["blockers"]:
        return 2
    if args.run:
        from src.experiment_runner import run_experiments
        run_experiments(args.config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
