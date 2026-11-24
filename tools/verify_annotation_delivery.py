"""独立检查两个实际ZIP：字节校验、公开字段、离线资源、A/B同批性。"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
from urllib.parse import unquote
import zipfile


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.embedded = [], []
        self.in_data = False

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        for key in ("src", "href"):
            if attr.get(key):
                self.links.append(attr[key])
        if tag == "script" and attr.get("id") == "package-data":
            self.in_data = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_data = False

    def handle_data(self, data):
        if self.in_data:
            self.embedded.append(data)


def verify(archives):
    checks, contents = [], []
    def check(name, ok):
        checks.append({"check": name, "passed": bool(ok)})
        if not ok:
            raise ValueError(f"交付检查失败：{name}")
    for slot, archive in zip(("A", "B"), archives):
        with zipfile.ZipFile(archive) as stream:
            check(f"{slot}:ZIP_CRC", stream.testzip() is None)
            names = stream.namelist()
            prefix = f"人工标注包_{slot}/"
            check(f"{slot}:固定顶层目录及无路径穿越", all(name.startswith(prefix) and ".." not in PurePosixPath(name).parts and not name.startswith("/") for name in names))
            check(f"{slot}:无重复ZIP路径", len(names) == len(set(names)))
            data = {name.removeprefix(prefix): stream.read(name) for name in names}
        check(f"{slot}:不附协调者私有/答案/权重文件", not any(any(word in name.lower() for word in ("private/", "claim_key", "reviewer_a", "reviewer_b", "train.json", "val.json", ".safetensors", "coordinator", "审查_仅协调者")) for name in data))
        load = lambda name: json.loads(data[name].decode("utf-8-sig"))
        lines = lambda name: [json.loads(line) for line in data[name].decode("utf-8-sig").splitlines() if line.strip()]
        manifest = load("package_manifest.json")
        check(f"{slot}:身份与未就绪状态", manifest["reviewer_slot"] == slot and not manifest["TRAINING_READY"] and not manifest["human_event_review_complete"])
        inventory = load("文件校验清单.json")["files"]
        check(f"{slot}:清单完整覆盖所有文件", {row["path"] for row in inventory} == set(data) - {"文件校验清单.json"})
        check(f"{slot}:全包SHA与大小", all(len(data[row["path"]]) == row["bytes"] and hashlib.sha256(data[row["path"]]).hexdigest() == row["sha256"] for row in inventory))
        claims, links, tasks = [lines("原始材料/" + name) for name in ("公开声明.jsonl", "候选关联.jsonl", "监督任务.jsonl")]
        images = load("原始材料/图像清单.json")
        check(f"{slot}:945声明187关联210模板1392图", tuple(map(len, (claims, links, tasks, images))) == (945, 187, 210, 1392))
        check(f"{slot}:声明仅公开白名单", all(set(row) == {"claim_id", "provisional_event_id", "usage", "dataset_partition", "claim_text", "claim_date", "claim_location", "images"} for row in claims))
        check(f"{slot}:候选关联仅公开白名单", all(set(row) == {"link_id", "event_group", "claim_a", "claim_b", "candidate_link_basis", "shared_claim_image_sha256"} for row in links))
        check(f"{slot}:监督任务未混入标签或数值特征", all(set(row) == {"task_id", "claim_id", "task_kind", "role", "event_group", "materials_ready", "material_version", "fixed_target_refs", "record_status", "training_ready"} and row["materials_ready"] is False and row["material_version"] is None and row["fixed_target_refs"] == [] and row["training_ready"] is False for row in tasks))
        check(f"{slot}:原图1393次引用完整且字节SHA匹配", sum(len(c["images"]) for c in claims) == 1393 and all(set(image) == {"image_id", "path", "sha256"} and image["path"] in data and hashlib.sha256(data[image["path"]]).hexdigest() == image["sha256"] for claim in claims for image in claim["images"]))
        check(f"{slot}:仅声明原图无多余档案图", {image["path"] for image in images} == {image["path"] for claim in claims for image in claim["images"]} == {name for name in data if name.startswith("原始材料/图像/")})
        check(f"{slot}:清单ID与数据相符", manifest["expected_claim_ids"] == [row["claim_id"] for row in claims] and manifest["expected_link_ids"] == [row["link_id"] for row in links] and manifest["expected_task_ids"] == [row["task_id"] for row in tasks])
        ids = set(manifest["expected_claim_ids"])
        check(f"{slot}:所有任务与关联引用已知声明", all(row["claim_a"] in ids and row["claim_b"] in ids for row in links) and all(row["claim_id"] in ids for row in tasks))
        check(f"{slot}:210监督仍待固定材料", len(manifest["task_materials_ready_by_id"]) == 210 and all(value is False for value in manifest["task_materials_ready_by_id"].values()))
        page = Page()
        page.feed(data["00_从这里开始.html"].decode("utf-8"))
        embedded = json.loads("".join(page.embedded))
        check(f"{slot}:工作台内嵌与原始材料一致", embedded == {"manifest": manifest, "claims": claims, "links": links, "tasks": tasks})
        html_files = {name: content for name, content in data.items() if name.endswith(".html")}
        offline_ok = True
        for name, content in html_files.items():
            parser = Page()
            parser.feed(content.decode("utf-8"))
            for link in parser.links:
                if link.startswith(("http:", "https:", "//", "file:")):
                    offline_ok = False
                    continue
                if link.startswith("#"):
                    continue
                target = str(PurePosixPath(name).parent / unquote(link.split("#")[0]))
                # 所有本次说明链接均使用同级路径，不依赖项目根或外部软件。
                if target not in data:
                    offline_ok = False
        check(f"{slot}:6份HTML离线链接全部存在", len(html_files) == 6 and offline_ok)
        check(f"{slot}:CSV含UTF8_BOM", data["原始材料/声明索引.csv"].startswith(b"\xef\xbb\xbf"))
        check(f"{slot}:无已填人工结果", not any(name.startswith("review_return_") for name in data))
        check(f"{slot}:教学文件显式DEMO不可训练", load("示例/返回结构_DEMO_ONLY_不要提交.json")["package_id"] == "DEMO_ONLY_NOT_FOR_TRAINING" and load("示例/训练结构_DEMO_ONLY.json")["DEMO_ONLY"] is True)
        contents.append(data)
    changed = {name for name in set(contents[0]) | set(contents[1]) if contents[0].get(name) != contents[1].get(name)}
    check("A/B公开材料原图与说明逐字一致", changed <= {"00_从这里开始.html", "package_manifest.json", "文件校验清单.json", "请先读我.txt", "示例/返回结构_DEMO_ONLY_不要提交.json"})
    return {"status": "VERIFIED", "checks": checks, "passed": len(checks), "failed": 0,
            "scope": "工程与交付完整性核查；不是人工标注、语义仲裁或训练完成证明", "human_event_review_complete": False, "TRAINING_READY": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify([args.a, args.b])
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed": report["passed"], "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
