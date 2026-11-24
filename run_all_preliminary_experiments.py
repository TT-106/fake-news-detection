"""本地数据核查与受控仿真入口；正式实验不再回退到人工生成的高分结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_integrity import AI_EVENT_PROTOCOL, provisional_split
from src.real_averimatec_pipeline import check_readiness


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="核查真实数据和实验准备状态，不下载模型、不调用付费 API")
    parser.add_argument("--synthetic", action="store_true", help="另运行明确标记为受控合成的 32 组公式仿真")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/preflight")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readiness = check_readiness()
    audit = readiness["dataset_audit"]
    if audit is None:
        for blocker in readiness["blockers"]:
            if blocker["component"] == "dataset_audit":
                print(f"数据核查缺少文件：{blocker['path']}")
    elif audit["protocol"] == AI_EVENT_PROTOCOL:
        (args.output_dir/"sealed_dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已核查锁定事件划分：{audit['isolation_status']}。终评文件未打开。")
    else:
        split = provisional_split(audit)
        (args.output_dir/"dataset_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output_dir/"provisional_split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"本地数据核查: train={audit['train']['count']}, val={audit['val']['count']}。")
        print(f"候选划分: {split['counts']}；待人工事件复核。")
    (args.output_dir/"preflight.json").write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.synthetic:
        from src.phase1_audit_and_fix import run_exp2_v2_non_circular
        directory = args.output_dir/"controlled_synthetic"
        directory.mkdir(parents=True, exist_ok=True)
        simulation = run_exp2_v2_non_circular(str(directory))
        simulation.update({"evidence_protocol": "CONTROLLED_SYNTHETIC_MECHANISM_ONLY",
                           "formal_multimodal_results": False,
                           "interpretation": "检验公式实现和指定输入上的机制；不能证明真实数据或相对基线的优势。"})
        (directory/"exp2_synthetic_32_v2_report.json").write_text(json.dumps(simulation, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"受控合成仿真: {simulation['num_controlled_configurations']} 组；不计作正式多模态实验。")
    print(f"正式实验状态: {readiness['status']}；未生成正式分数。")
    return 2 if readiness["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
