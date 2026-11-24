"""取得公开研究材料并锁定上游版本；不生成标注，不读取评测答案用于推理。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATA_REPO = "Rui4416/AVerImaTeC"
DATA_REVISION = "7e77fb26a1ba86159ff5ba27b59f373b450d6fd0"
TRAIN_SHA256 = "0f0d2ff69b08a57439ec7ea532336150069c5534880768f96def92bea95df1f1"


def acquire(kind):
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "90")
    # 当前 Windows 网络已实测 HTTP Range 可用；避免 CAS/Xet 长时间无字节进展。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=False)
    if kind == "model":
        repo, repo_type = MODEL_REPO, "model"
        info = api.model_info(repo, revision=MODEL_REVISION, files_metadata=True)
        directory = ROOT / "models/Qwen2.5-VL-7B-Instruct"
        selected = [f for f in info.siblings if f.rfilename.endswith((".json", ".jinja", ".txt", ".safetensors")) or f.rfilename == "README.md"]
    else:
        repo, repo_type = DATA_REPO, "dataset"
        info = api.dataset_info(repo, revision=DATA_REVISION, files_metadata=True)
        directory = ROOT / "data/averimatec/upstream"
        names = {"images.zip", "README.md"}
        if kind == "dataset":
            names.add("train.json")
        selected = [f for f in info.siblings if f.rfilename in names]
        if {f.rfilename for f in selected} != names:
            raise RuntimeError("固定的数据集版本缺少所需文件")
        train = directory.parent / "train.json"
        if kind == "dataset" and train.exists() and hashlib.sha256(train.read_bytes()).hexdigest() != TRAIN_SHA256:
            raise RuntimeError(f"现有训练文件与固定版本不一致：{train}")
    directory.mkdir(parents=True, exist_ok=True)
    lock = {"repo_id": repo, "repo_type": repo_type, "revision": info.sha, "started_at": datetime.now(timezone.utc).isoformat(), "files": []}
    for item in selected:
        lfs = item.lfs
        expected = lfs.sha256 if lfs else None
        lock["files"].append({"path": item.rfilename, "size": item.size, "lfs_sha256": expected})
    (directory / "UPSTREAM_LOCK.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")

    def download(entry):
        path = Path(hf_hub_download(repo, entry["path"], repo_type=repo_type, revision=info.sha,
                                   local_dir=directory, token=False))
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if entry["lfs_sha256"] and actual != entry["lfs_sha256"]:
            raise RuntimeError(f"上游 LFS SHA-256 不匹配：{entry['path']}")
        if entry["path"] == "train.json" and actual != TRAIN_SHA256:
            raise RuntimeError("训练 JSON 的 SHA-256 与固定版本不匹配")
        print(f"已校验 {entry['path']} ({path.stat().st_size} bytes)", flush=True)
        return dict(entry, sha256=actual, verified_size=path.stat().st_size)

    with ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(download, lock["files"]))
    lock.update(files=files, finished_at=datetime.now(timezone.utc).isoformat(), status="HASH_VERIFIED")
    (directory / "UPSTREAM_LOCK.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    if kind != "model":
        from PIL import Image
        destination = ROOT / "data/averimatec/images"
        destination.mkdir(parents=True, exist_ok=True)
        manifest = []
        with zipfile.ZipFile(directory / "images.zip") as z:
            for item in z.infolist():
                if item.is_dir() or not item.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                # 原图用官方文件名映射；拒绝歧义/冲突，不覆盖不同内容。
                name = Path(item.filename).name
                raw = z.read(item)
                sha = hashlib.sha256(raw).hexdigest()
                target = destination / name
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                    raise RuntimeError(f"重复文件名有不同内容：{name}")
                target.write_bytes(raw)
                with Image.open(target) as image:
                    image.verify()
                manifest.append({"archive_member": item.filename, "path": target.relative_to(ROOT).as_posix(), "sha256": sha, "bytes": len(raw)})
        (destination / "image_manifest.json").write_text(json.dumps({"upstream_revision": info.sha, "images": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已恢复并实际解码校验 {len(manifest)} 张原图", flush=True)
        if kind == "dataset" and not train.exists():
            with train.open("xb") as stream:
                stream.write((directory / "train.json").read_bytes())
        if kind == "dataset":
            print(f"训练集：{train}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["model", "dataset", "images"],
                        help="model 下载 Qwen 权重；dataset 下载训练 JSON 和原图；images 只下载原图")
    acquire(parser.parse_args().kind)
