"""真实 Qwen 的训练集先导对照；不触碰锁定前的官方 152 条评测集。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.core_models import LABELS
from src.input_views import InputView
from src.research_contracts import CostLedger, ImageAsset, ProtocolError, PublicClaim

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selected_public_claims(config, root=ROOT):
    """在任何推理调用前物理构造白名单对象；不返回真值或核查 QA。"""
    root = Path(root)
    split = load_json(root / config["split_path"])
    if split.get("evaluation_scope") != "TRAIN_PILOT_ONLY_NOT_FINAL_EVAL":
        raise ProtocolError("本入口仅允许事先固定的训练集先导协议")
    source = root / "data/averimatec/train.json"
    if sha(source) != split["dataset_sha256"]:
        raise ProtocolError("先导数据版本改变")
    ids = split.get("claim_ids", [])
    if not ids or len(ids) != len(set(ids)) or any(not str(cid).startswith("train:") for cid in ids):
        raise ProtocolError("先导声明必须是唯一训练集 ID，禁止使用官方 val/final_eval")
    source_split = root / split["source_split"]
    if sha(source_split) != split["source_split_sha256"]:
        raise ProtocolError("原候选先导划分版本改变；不能事后重选样本")
    allowed = load_json(source_split)["split"]["pilot"]
    if set(ids) != set(allowed):
        raise ProtocolError("必须使用完整事前先导列表，不能按预测或真值筛选样本")
    samples = load_json(source)
    manifest = load_json(root / "data/averimatec/images/image_manifest.json")
    image_index = {Path(row["path"]).name: row for row in manifest["images"]}
    claims = []
    for cid in ids:
        sample = samples[int(cid.split(":")[1])]
        assets = []
        for filename in sample.get("claim_images", []):
            if filename not in image_index:
                raise ProtocolError(f"声明原图缺失：{filename}")
            row = image_index[filename]
            asset = ImageAsset(str(root / row["path"]), row["sha256"])
            asset.verify()
            assets.append(asset)
        if not assets:
            raise ProtocolError(f"多模态先导声明缺少原图：{cid}")
        claim = PublicClaim.from_sample(sample, cid, split["event_groups"][cid], tuple(assets))
        claims.append(claim)
    return claims


def evaluate_saved_predictions(predictions_path, split_path, root=ROOT):
    """预测关闭后单独读取真值；四类 Macro-F1 的类别集合固定。"""
    from src.research_evaluation import classification_metrics
    root = Path(root)
    split = load_json(split_path)
    if (split.get("evaluation_scope") != "TRAIN_PILOT_ONLY_NOT_FINAL_EVAL"
            or sha(root / "data/averimatec/train.json") != split["dataset_sha256"]
            or sha(root / split["source_split"]) != split["source_split_sha256"]):
        raise ProtocolError("评分时数据或先导划分版本改变，拒绝发布成绩")
    rows = [json.loads(line) for line in Path(predictions_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row["id"] for row in rows] != split["claim_ids"]:
        raise ProtocolError("预测必须覆盖完整先导样本且顺序/身份一致")
    train = load_json(root / "data/averimatec/train.json")
    actual = [train[int(row["id"].split(":")[1])]["label"] for row in rows]
    predicted = [row["prediction"] for row in rows]
    return classification_metrics(actual, predicted)


def run(config, root=ROOT):
    root = Path(root)
    frozen_split_sha256 = sha(root / config["split_path"])
    claims = selected_public_claims(config, root)
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.jsonl"
    report_path = output / "report.json"
    if predictions_path.exists() or report_path.exists():
        raise ProtocolError("输出目录已有运行产物；每次运行必须使用独立目录")
    mode = config.get("input_mode", "image_text")
    if mode not in {"image_text", "text_only"}:
        raise ProtocolError("未知先导输入模式")
    import torch
    from src.vlm_backend import LocalQwenVLM
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    started = time.perf_counter()
    model = LocalQwenVLM(str(root / config["model_dir"]), config["quantization"], str(output / "calls"),
                         max_pixels=config["max_pixels"], max_new_tokens=config["max_new_tokens"])
    if mode == "text_only":
        model = model.with_view(InputView("no_image"))
    rows = []
    with predictions_path.open("x", encoding="utf-8") as stream:
        for claim in claims:
            ledger = CostLedger()
            begin = time.perf_counter()
            # 直接零检索对照；不伪造事实片段、来源关系、充分性或人工规则。
            payload = {"claim": claim.text, "atoms": [], "evidence": [], "z": [],
                       "gaps": ["No external evidence has been acquired in this zero-retrieval baseline."]}
            values = model._call("terminal", claim, payload, ledger, logits=True)
            logits = np.asarray(values, dtype=float)
            if logits.shape != (4,) or not np.all(np.isfinite(logits)):
                raise ProtocolError("必须由真实 forward 取得四个有限标签 logits")
            label = LABELS[int(np.argmax(logits))]
            row = {"id": claim.claim_id, "prediction": label, "logits": logits.tolist(),
                   "image_sha256": [asset.sha256 for asset in claim.images],
                   "input_mode": mode, "cost_rmb": ledger.total,
                   "elapsed_seconds": time.perf_counter()-begin,
                   "requests": [entry.call_id for entry in ledger.entries]}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            stream.flush()
            rows.append(row)
            print(f"先导完成 {len(rows)}/{len(claims)} {claim.claim_id} -> {label}", flush=True)
    if sha(root/config["split_path"]) != frozen_split_sha256:
        raise ProtocolError("运行期间先导清单被修改，拒绝发布成绩")
    evaluation = evaluate_saved_predictions(predictions_path, root/config["split_path"], root)
    metrics = {"macro_f1": evaluation["macro_f1"], "accuracy": evaluation["accuracy"], "coverage": 1.,
               "mean_cost_rmb": sum(r["cost_rmb"] for r in rows)/len(rows),
               "latency_seconds": sum(r["elapsed_seconds"] for r in rows)/len(rows)}
    report = {"n_samples": len(rows), "metrics": metrics, "per_class": evaluation["per_class"],
              "evaluation_scope": "TRAIN_PILOT_ONLY_NOT_FINAL_EVAL", "formal_e1_e4_results": False,
              "method": "DIRECT_FROZEN_QWEN_ZERO_RETRIEVAL", "input_mode": mode,
              "human_event_review_complete": False, "human_supervision_used": False,
              "model_identity": asdict(model.identity), "config": config,
              "split_sha256": frozen_split_sha256, "sample_ids": [r["id"] for r in rows],
              "prediction_sha256": sha(predictions_path), "total_elapsed_seconds": time.perf_counter()-started,
              "finished_at": datetime.now(timezone.utc).isoformat(),
              "interpretation": "真实模型先导对照，未使用外部取证或充分性门控；不能替代正式E1—E4、论文复现或有据正确回答评价。"}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "metrics": metrics}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(load_json(args.config))
