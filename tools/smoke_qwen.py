"""真实原图 + 冻结 Qwen 7B 4bit 冒烟；不计算检测 F1。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pilot_vlm import load_json, selected_public_claims
from src.research_contracts import CostLedger, ProtocolError


def main(config):
    import numpy as np
    import torch
    from src.core_models import LABELS
    from src.vlm_backend import LocalQwenVLM
    model_dir = ROOT / config["model_dir"]
    lock = load_json(model_dir / "UPSTREAM_LOCK.json")
    if lock.get("status") != "HASH_VERIFIED":
        raise ProtocolError("完整权重尚未通过上游SHA校验，不能启动冒烟")
    for item in lock["files"]:
        path = model_dir / item["path"]
        if not path.is_file() or path.stat().st_size != item["verified_size"]:
            raise ProtocolError("模型锁定文件缺失或长度改变")
        if item.get("lfs_sha256"):
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != item["lfs_sha256"]:
                raise ProtocolError("权重与上游LFS SHA不匹配")
    pilot_config = load_json(ROOT / config["pilot_config"])
    public = selected_public_claims(pilot_config)
    claim = next(c for c in public if c.claim_id == config["claim_id"])
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("predictions.jsonl", "report.json")):
        raise ProtocolError("冒烟输出已存在，请使用新运行目录")
    torch.manual_seed(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    print("加载经LFS SHA校验的官方Qwen权重", flush=True)
    model = LocalQwenVLM(str(model_dir), config["quantization"], str(output/"calls"),
                         max_pixels=config["max_pixels"], max_new_tokens=config["max_new_tokens"])
    import bitsandbytes as bnb
    import importlib.metadata
    quantized = [module for module in model.model.modules() if isinstance(module, bnb.nn.Linear4bit)]
    quantized_states = sum(module.weight.dtype == torch.uint8 and getattr(module.weight, "quant_state", None) is not None
                           for module in quantized)
    if (not getattr(model.model, "is_loaded_in_4bit", False)
            or not quantized or quantized_states != len(quantized)):
        raise ProtocolError("未实际取得4bit量化层/uint8权重与量化状态，拒绝仅凭配置宣称成功")
    print("完整模型已加载，执行真实原图分析与标签forward", flush=True)
    ledger = CostLedger()
    analysis = model.analyse(claim, ledger)
    logits = model._call("terminal", claim,
                         {"claim": claim.text, "atoms": [asdict(a) for a in analysis.atoms],
                          "evidence": [], "z": [], "gaps": analysis.gaps}, ledger, logits=True)
    if len(logits) != 4 or not np.all(np.isfinite(logits)):
        raise ProtocolError("标签forward未返回四个有限真实logits")
    if not all(not p.requires_grad for p in model.model.parameters()):
        raise ProtocolError("模型权重未全部冻结")
    actual_quant = getattr(model.model.config, "quantization_config", None)
    actual_quant = actual_quant.to_dict() if hasattr(actual_quant, "to_dict") else actual_quant
    elapsed = time.perf_counter()-began
    row = {"id": claim.claim_id, "prediction": LABELS[int(np.argmax(logits))], "logits": logits}
    (output/"predictions.jsonl").write_text(json.dumps(row, ensure_ascii=False)+"\n", encoding="utf-8")
    report = {"n_samples": 1, "metrics": {"latency_seconds": elapsed},
              "status": "REAL_IMAGE_ANALYSIS_AND_LOGITS_PASSED", "experiment_stage": "engineering",
              "formal_e1_e4_results": False, "classification_f1": None,
              "claim_id": claim.claim_id, "image_sha256": [a.sha256 for a in claim.images],
              "num_atoms": len(analysis.atoms), "num_rules": len(analysis.rules),
              "rule_scope": "No human rule catalogue supplied; no generic rules added",
              "model_identity": asdict(model.identity), "actual_quantization": actual_quant,
              "is_loaded_in_4bit": model.model.is_loaded_in_4bit,
              "linear4bit_modules": len(quantized), "verified_uint8_quant_states": quantized_states,
              "runtime_python": sys.executable,
              "runtime_packages": {name: importlib.metadata.version(name) for name in
                                   ("torch", "transformers", "bitsandbytes", "accelerate")},
              "device_map": getattr(model.model, "hf_device_map", {}),
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "model_memory_footprint_bytes": model.model.get_memory_footprint(),
              "cost_entries": [asdict(e) for e in ledger.entries], "elapsed_seconds_including_load": elapsed,
              "notes": "只验证真实模型可加载、可读原图并前向；未使用QA/真值，不计算正式分类性能。"}
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    main(load_json(parser.parse_args().config))
