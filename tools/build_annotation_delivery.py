"""把待审核模板打成两份可离线使用的独立人工包；不生成标签或训练数据。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from validate_annotation_return import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/annotations/pending/review-20261117-01"
AUDIT = ROOT / "outputs/annotation_delivery_20261117"
DOC_NAMES = ["00_快速开始.html", "01_分任务操作清单.html", "02_逐列填写说明.html",
             "03_最终返还清单与训练衔接.html", "04_数据与训练格式.html"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def safe_csv(value):
    text = str(value if value is not None else "")
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows([[safe_csv(item) for item in row] for row in rows])


def public_source(reviewer):
    base = SOURCE / "event_review" / reviewer
    return (read_lines(base / "claims.jsonl"), read_lines(base / "candidate_links.jsonl"),
            read_lines(SOURCE / "annotation_review" / reviewer / "tasks.jsonl"))


def gather_material():
    original_claims, original_links, original_tasks = public_source("reviewer_a")
    if (original_claims, original_links, original_tasks) != public_source("reviewer_b"):
        raise ValueError("A/B源模板不一致，须先确认同批独立审核范围")
    files = {}
    for claim in original_claims:
        for item in claim["claim_images"]:
            relative = item["file"]
            source = (ROOT / relative).resolve()
            source.relative_to((ROOT / "data/averimatec/images").resolve())
            if not source.is_file() or sha256(source) != item["sha256"]:
                raise ValueError(f"声明原图缺失或SHA不匹配：{relative}")
            if relative in files and files[relative]["sha256"] != item["sha256"]:
                raise ValueError("相同文件的SHA声明相冲突")
            files[relative] = {"source": source, "sha256": item["sha256"]}
    images = []
    for index, (relative, entry) in enumerate(sorted(files.items()), 1):
        image_id = f"I{index:06d}"
        suffix = entry["source"].suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError(f"未支持的图像类型：{suffix}")
        entry.update(image_id=image_id, path=f"原始材料/图像/{image_id}{suffix}")
        images.append({key: entry[key] for key in ("image_id", "path", "sha256")})
    claims = [{"claim_id": row["claim_id"], "provisional_event_id": row["provisional_event_id"],
               "usage": row["usage_boundary"], "dataset_partition": row["dataset_partition"],
               "claim_text": row["claim_text"], "claim_date": row["claim_date"],
               "claim_location": row["claim_location"],
               "images": [{key: files[image["file"]][key] for key in ("image_id", "path", "sha256")}
                          for image in row["claim_images"]]} for row in original_claims]
    links = [{"link_id": f"L{index:06d}", **{key: row[key] for key in (
        "event_group", "claim_a", "claim_b", "candidate_link_basis", "shared_claim_image_sha256")}}
             for index, row in enumerate(original_links, 1)]
    tasks = [{"task_id": row["task_id"], "claim_id": row["claim_id"], "task_kind": row["task_kind"],
              "role": row["role"], "event_group": row["event_group"], "materials_ready": False,
              "material_version": None, "fixed_target_refs": [],
              "record_status": "PENDING_MATERIALS", "training_ready": False}
             for row in original_tasks]
    if (len(claims), len(links), len(tasks), len(images)) != (945, 187, 210, 1392):
        raise ValueError("数据范围发生变化，必须同步更改本批规程与说明")
    return claims, links, tasks, images, files


def build(output, package_id, archive_suffix=""):
    if not re.fullmatch(r"annotation-\d{8}-v\d+", package_id):
        raise ValueError("package-id格式须为annotation-YYYYMMDD-vN")
    if archive_suffix and not re.fullmatch(r"[\w-]+", archive_suffix):
        raise ValueError("archive-suffix只能使用文字、数字、下划线或连字符")
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已存在的交付目录：{output}；请使用新目录/版本")
    doc_dir = AUDIT / "docs_draft"
    for name in DOC_NAMES:
        if not (doc_dir / name).is_file():
            raise FileNotFoundError(f"说明尚未完成：{name}")
    training_demo = doc_dir / "训练结构_DEMO_ONLY.json"
    if not training_demo.is_file():
        raise FileNotFoundError("训练结构示例尚未完成")
    claims, links, tasks, images, files = gather_material()
    upstream = read_json(ROOT / "data/averimatec/upstream/UPSTREAM_LOCK.json")
    template = (ROOT / "tools/annotation_workbench.template.html").read_text(encoding="utf-8")
    if template.count("__ANNOTATION_DATA_JSON__") != 1:
        raise ValueError("工作台模板占位符错误")
    # 在预检完成后创建；失败时保留现场，绝不清理或覆盖用户结果。
    output.mkdir(parents=True)
    audit_dir = AUDIT / "coordinator"
    audit_dir.mkdir(exist_ok=True)
    private_key = audit_dir / "claim_key.json"
    if not private_key.exists():
        shutil.copy2(SOURCE / "private/claim_key.json", private_key)
    archives = []
    for slot in ("A", "B"):
        folder_name = f"人工标注包_{slot}"
        folder = output / folder_name
        (folder / "原始材料/图像").mkdir(parents=True)
        (folder / "说明").mkdir()
        (folder / "示例").mkdir()
        manifest = {
            "schema_version": 1, "package_id": package_id, "reviewer_slot": slot,
            "expected_claim_ids": [row["claim_id"] for row in claims],
            "expected_link_ids": [row["link_id"] for row in links],
            "expected_task_ids": [row["task_id"] for row in tasks],
            "task_kind_by_id": {row["task_id"]: row["task_kind"] for row in tasks},
            "task_materials_ready_by_id": {row["task_id"]: False for row in tasks},
            "counts": {"claims": len(claims), "candidate_groups": len({row["provisional_event_id"] for row in claims}),
                       "candidate_links": len(links), "supervision_templates": len(tasks),
                       "image_references": sum(len(row["images"]) for row in claims), "original_image_files": len(images)},
            "source": {"dataset": "AVerImaTeC", "repo_id": upstream["repo_id"],
                       "image_archive_revision": upstream["revision"],
                       "train_json_sha256": sha256(ROOT / "data/averimatec/train.json"),
                       "val_json_sha256": sha256(ROOT / "data/averimatec/val.json")},
            "protocol": "independent_event_asset_review_v1_pending_fixed_supervision_materials",
            "human_event_review_complete": False, "TRAINING_READY": False,
            "answers_included": False, "other_reviewer_results_included": False,
        }
        validate_manifest(manifest)
        write_json(folder / "package_manifest.json", manifest)
        for entry in files.values():
            shutil.copyfile(entry["source"], folder / entry["path"])
        write_lines(folder / "原始材料/公开声明.jsonl", claims)
        write_lines(folder / "原始材料/候选关联.jsonl", links)
        write_lines(folder / "原始材料/监督任务.jsonl", tasks)
        write_json(folder / "原始材料/图像清单.json", images)
        write_csv(folder / "原始材料/声明索引.csv",
                  ["声明编号", "候选隔离组", "用途边界", "原始分区", "声明原文", "声明发布日期(非拍摄日)", "地点", "图像编号", "图像相对路径"],
                  [[row["claim_id"], row["provisional_event_id"], row["usage"], row["dataset_partition"],
                    row["claim_text"], row["claim_date"], row["claim_location"],
                    " | ".join(image["image_id"] for image in row["images"]),
                    " | ".join(image["path"] for image in row["images"])] for row in claims])
        for name in DOC_NAMES:
            shutil.copyfile(doc_dir / name, folder / "说明" / name)
        shutil.copyfile(training_demo, folder / "示例/训练结构_DEMO_ONLY.json")
        embedded = json.dumps({"manifest": manifest, "claims": claims, "links": links, "tasks": tasks}, ensure_ascii=False, separators=(",", ":"))
        for char, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
            embedded = embedded.replace(char, escaped)
        (folder / "00_从这里开始.html").write_text(template.replace("__ANNOTATION_DATA_JSON__", embedded), encoding="utf-8")
        (folder / "请先读我.txt").write_text(
            f"人工标注工作台 {slot} 包\n\n"
            "1. 右键 ZIP，选「全部解压缩」。从解压后的目录打开 00_从这里开始.html，填自己的署名。\n"
            "2. 第②页检查 945 条图文。看配文和图片，查同组和其他组，选怎么分组，写理由并保存。\n"
            "3. 第③页对比 187 对图文。选两条的关系，写理由，点「保存本对」。本轮先跳过第④页。\n"
            "4. 到第⑤页看进度，点「导出审核结果 JSON」。只交回自己最新的 review_return_*.json。\n\n"
            "A、B 两人各自看完整的同一批图文，独立填写。每一步的示例见「说明」文件夹。\n"
            "原图打不开时，请确认保留了「原始材料/图像」文件夹。\n"
            "旧版结果可直接导入同字母的新包。只有浏览器草稿时，请先从旧页面导出，再导入新版。\n", encoding="utf-8-sig")
        demo = {"schema_version": 1, "package_id": "DEMO_ONLY_NOT_FOR_TRAINING", "reviewer_slot": slot,
                "reviewer_name": "教学示例（请勿提交）",
                "event_reviews": [{"claim_id": "DEMO_C1", "decision": "change", "proposed_event_id": "", "merge_with": ["DEMO_C2"],
                                   "reason": "教学：两条配文不同，但可见同一原始照片的裁剪，建议按素材关联隔离。", "images_reviewed": True, "cross_group_checked": True}],
                "link_reviews": [{"link_id": "DEMO_L1", "relation": "same_asset_only", "reason": "教学：人物位置与背景构图一致，是同素材的裁剪版本。"}],
                "annotation_reviews": [{"task_id": "DEMO_SOURCE1", "status": "pending_materials", "label": None, "class_label": None,
                                        "reason": "未收到同一组固定证据A/B及页面快照，待协调者补齐。", "evidence_refs": []}]}
        write_json(folder / "示例/返回结构_DEMO_ONLY_不要提交.json", demo)
        inventory = [{"path": file.relative_to(folder).as_posix(), "bytes": file.stat().st_size, "sha256": sha256(file)}
                     for file in sorted(folder.rglob("*")) if file.is_file()]
        write_json(folder / "文件校验清单.json", {"excluded_self": "文件校验清单.json", "files": inventory})
        suffix = f"_{archive_suffix}" if archive_suffix else ""
        archive = output / f"{folder_name}_20261117{suffix}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
            for file in sorted(folder.rglob("*")):
                if file.is_file():
                    stream.write(file, f"{folder_name}/{file.relative_to(folder).as_posix()}")
        with zipfile.ZipFile(archive) as stream:
            error = stream.testzip()
            if error:
                raise ValueError(f"ZIP CRC失败：{error}")
        archives.append({"reviewer_slot": slot, "path": archive.relative_to(ROOT).as_posix(), "bytes": archive.stat().st_size,
                         "sha256": sha256(archive), "file_count": len(inventory) + 1})
    report = {"package_id": package_id, "archives": archives, "counts": manifest["counts"],
              "all_supervision_materials_ready": False, "human_event_review_complete": False, "TRAINING_READY": False,
              "private_mapping_excluded": True, "answers_excluded": True}
    write_json(output / "交付校验报告.json", report)
    guide = """# 协调者交接说明

给甲：人工标注包_A_20261117.zip；给乙：人工标注包_B_20261117.zip。两人审同一批，不各分半，不互看判断。
完整解压后打开00_从这里开始.html；只需交回各自最新review_return_*.json。可以分批，不填的仍是未完成。
当前能开展945条事件/素材审核、187条候选关联。210个监督模板尚缺固定对象，不是已可标的210条训练样本。

## 回收验收

在项目根目录运行（下列文件名用收到的实际路径替换；Python只由协调者使用）：

```powershell
python tools/validate_annotation_return.py validate --file '收到的A.json' --manifest 'A包目录/package_manifest.json'
python tools/validate_annotation_return.py validate --file '收到的B.json' --manifest 'B包目录/package_manifest.json'
python tools/validate_annotation_return.py compare --a '收到的A.json' --b '收到的B.json' --manifest-a 'A包目录/package_manifest.json' --manifest-b 'B包目录/package_manifest.json' --out '新的分歧审查目录'
```

验收器核对版本、A/B编号、本人署名、已知ID、字段、实际填写状态；不补标签，不进行训练。它只能验证记录，不能认证人是否真的独立。
compare列出缺项、两人一致项、分歧和材料引用冲突；一致项也不自动当作已仲裁训练数据。
第三位不同人员处理分歧并写依据；未提供证据不是负例，实际审核unknown也不能等同于未开始。

## 从人工结果到训练

1. claim_key.json是私有的C/G到原始train:id/val:id映射，不给审核者、不进ZIP。
2. 合并事件与素材的隔离关联，复核漏合并和误合并；任一共享事件/素材连通分量不得跨用途泄漏。官方152仅事件复核及最终评测，不能进入拟合、阈值调优或校准。
3. 在正式标签审核前，补齐来源证据对、atom×passage、实际history+selected_label+citations的固定对象/版本与可定位快照；两人看到同一对象。一个模板若展开多个对象，分配新的task_id并升级包版本，保留原始返回。
4. 由真实程序生成6/3/7维features、SHA与边界、真实模型logits。由审核者填写判断、理由、引用，不能人工编数或复用数据集真假标签作为三类监督。
5. 经有效双人标签/必要第三人仲裁及证据验真后，协调者转换成说明/04中的训练JSONL，运行原有数据验证；该转换尚未自动实现，不可把review_return.json直接传给训练器。
6. 校准辅助标签不是当前runner的直接输入；温度拟合仍须锁定开发划分上的真实logits和类别标签。行动训练还需真实分支轨迹及成本/损失差。

交付、验收通过、两份一致都不能自动将TRAINING_READY/human_event_review_complete设为true。
文件校验清单列每包文件SHA；根交付校验报告列两个ZIP的SHA与大小。最终人工结果要另存，不覆盖底稿。
"""
    if not archive_suffix:
        (audit_dir / "协调者交接说明.md").write_text(guide, encoding="utf-8")
        write_json(AUDIT / "delivery_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "deliverables/annotation-20261117-v1")
    parser.add_argument("--package-id", default="annotation-20261117-v1")
    parser.add_argument("--archive-suffix", default="", help="ZIP 文件名后缀，保留原回收批次编号")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.package_id, args.archive_suffix), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
