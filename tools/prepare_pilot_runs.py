"""在权重完整校验后生成冒烟与先导的独立执行清单；不执行模型。"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.research_contracts import fingerprint
from src.vlm_backend import BNB4_LOADING_SETTINGS, PROCESSOR_LOADING_SETTINGS, TERMINAL_LOGITS_SETTINGS, PROMPTS


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    lock = load(ROOT / "models/Qwen2.5-VL-7B-Instruct/UPSTREAM_LOCK.json")
    if lock.get("status") != "HASH_VERIFIED" or any(not x.get("sha256") for x in lock["files"]):
        raise ValueError("模型尚未完成上游哈希校验，不生成可执行清单")
    python = str(ROOT / ".venv-research/Scripts/python.exe")
    source_split = ROOT / "config/pilot_split_20261117.json"
    first = load(source_split)["claim_ids"][0]
    smoke_split = ROOT / "config/model_smoke_split_20261117.json"
    smoke_split.write_text(json.dumps({"scope": "ENGINEERING_FIRST_PILOT_INPUT_ONLY", "claim_ids": [first],
                                      "parent_split": source_split.relative_to(ROOT).as_posix(),
                                      "parent_split_sha256": digest(source_split)}, ensure_ascii=False, indent=2), encoding="utf-8")
    entries = [("qwen-smoke-v1", "engineering", "method", "engineering_qwen_smoke", "完整Qwen真实原图冒烟", "config/model_smoke_qwen_v1.json", smoke_split, 1)]
    entries += [("qwen-direct-"+mode+"-v1", "pilot", "baseline", "averimatec-pilot20",
                 "Qwen图文零检索先导" if mode == "image_text" else "Qwen纯文本零检索先导",
                 f"config/pilot_qwen_direct_{mode}_v1.json", source_split, 20)
                for mode in ("image_text", "text_only")]
    target = ROOT / "progress/manifests"
    target.mkdir(parents=True, exist_ok=True)
    for rid,stage,kind,dataset,title,cpath,split,n in entries:
        config = load(ROOT / cpath)
        resource_identity = {row["path"]: row["sha256"] for row in lock["files"]}
        model_id = fingerprint({"repo": lock["repo_id"], "revision": lock["revision"], "resources": resource_identity,
                                "quantization": BNB4_LOADING_SETTINGS, "processor": PROCESSOR_LOADING_SETTINGS,
                                "terminal_logits": TERMINAL_LOGITS_SETTINGS,
                                "prompts": PROMPTS, "max_pixels": config["max_pixels"], "max_new_tokens": config["max_new_tokens"]})
        output = config["output_dir"]
        command = [python, "-X", "utf8"] + (["tools/smoke_qwen.py"] if stage == "engineering" else ["-m", "src.pilot_vlm"])
        command += ["--config", cpath]
        run = {"schema_version": 1, "id": rid, "version": "v1", "title": title, "status": "running", "kind": kind,
               "experiment_stage": stage, "method_id": "qwen_smoke" if stage == "engineering" else "direct_vlm_"+config["input_mode"],
               "series_id": "qwen_smoke" if stage == "engineering" else "direct_vlm_"+config["input_mode"],
               "review_status": "unreviewed", "dataset_id": dataset,
               "dataset_revision": load(source_split)["dataset_sha256"],
               "split_id": "pilot-first-input-smoke-only" if stage == "engineering" else "train-pilot20-20261117-unreviewed-events",
               "split_sha256": digest(split), "protocol_id": "QWEN_ENGINEERING_SMOKE_V1" if stage == "engineering" else "QWEN_TRAIN_PILOT_DIRECT_V1",
               "model_id": model_id, "budget_id": "zero-external-tool-calls", "n_samples": n, "seed": config["seed"],
               "started_at": datetime.now(timezone.utc).isoformat(), "command": command, "config_path": cpath, "metrics": {},
               "artifacts": {"report": output+"/report.json", "predictions": output+"/predictions.jsonl",
                             "config": cpath, "split": split.relative_to(ROOT).as_posix()}}
        path = target / (rid+".json")
        if path.exists():
            raise ValueError(f"执行清单已存在，拒绝覆盖：{path}")
        path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
