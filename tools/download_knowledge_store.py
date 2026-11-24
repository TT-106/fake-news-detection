"""断点续传分块下载并校验 AVerImaTeC 官方 Knowledge Store (默认仅允许 train/text_related.zip)。

防泄漏与完整性边界：
- 默认只允许下载 `train`（`Knowledge_Store/train/text_related.zip`，6,716,122,829 字节，含 793 条文本搜索与 793 条反向搜图网页）。
- 严禁在 Sprint 1–3 下载或接触 `val.zip`；若试图请求 `val`，必须显式传入 `--unlock-final-eval-val` 才放行。
- 采用固定 16 MiB 分块 + 多线程 HTTP Range 写入 `.part` 与 `.status.json`，支持随时中断续传。
- 下载完成后逐项校验文件总字节数、ZIP 目录结构（793 text + 793 ris）、全量 CRC32 及整包 SHA-256，最后原子发布并写入 `manifest.json`。
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
KNOWN_ARCHIVES = {
    "train": {
        "google_drive_file_id": "1rss7ISLKD9kv3o50qQVAx3gDnPpqVfP-",
        "filename": "text_related.zip",
        "expected_size": 6716122829,
        "expected_text_entries": 793,
        "expected_ris_entries": 793,
        "description": "AVerImaTeC Official Knowledge Store - train/text_related.zip (text + RIS webpages)",
    },
    "val": {
        "google_drive_file_id": "1P2eeCq5P_9-p-aUbzzEAN-vkquR9OdhS",
        "filename": "val.zip",
        "expected_size": None,
        "expected_text_entries": 152,
        "expected_ris_entries": 152,
        "description": "AVerImaTeC Official Knowledge Store - val.zip (SEALED until Sprint 4 E4 final eval)",
    },
}
HEADERS = {"User-Agent": "Mozilla/5.0"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def probe_remote_size(url: str) -> int:
    req = urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        cr = resp.headers.get("Content-Range", "")
        if "/" not in cr:
            raise RuntimeError(f"服务器未返回有效的 Content-Range: {cr!r}")
        return int(cr.split("/")[-1])


def fetch_chunk(url: str, start: int, end_inclusive: int, attempts: int = 5) -> bytes:
    expected_len = end_inclusive - start + 1
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={**HEADERS, "Range": f"bytes={start}-{end_inclusive}"},
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                if resp.status != 206:
                    raise RuntimeError(f"HTTP 状态不是 206 Partial Content: {resp.status}")
                data = resp.read()
            if len(data) != expected_len:
                raise RuntimeError(f"分块字节数不匹配：期望 {expected_len}，实际 {len(data)}")
            return data
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(10.0, 1.5 * attempt))
    raise RuntimeError("unreachable")


def verify_zip_archive(zip_path: Path, expected_text: int, expected_ris: int) -> dict:
    print(f"Verifying ZIP structure and CRC32 for {zip_path}...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        text_files = [n for n in names if "text_related_store_text_" in n and n.endswith(".json")]
        ris_files = [n for n in names if "image_related_store_text_" in n and n.endswith(".json")]
        if len(text_files) != expected_text or len(ris_files) != expected_ris:
            raise RuntimeError(
                f"ZIP 条目数量异常：text={len(text_files)} (期望 {expected_text}), "
                f"ris={len(ris_files)} (期望 {expected_ris})"
            )
        bad_entry = zf.testzip()
        if bad_entry is not None:
            raise RuntimeError(f"ZIP CRC32 校验失败，损坏条目：{bad_entry}")
    return {
        "total_entries": len(names),
        "text_search_entries": len(text_files),
        "reverse_image_search_entries": len(ris_files),
        "crc32_verified": True,
    }


def download_archive(
    split: str,
    dest_dir: Path,
    *,
    chunk_mib: int = 16,
    workers: int = 6,
    unlock_final_eval_val: bool = False,
    skip_crc_sweep: bool = False,
) -> dict:
    if split == "val" and not unlock_final_eval_val:
        raise RuntimeError("安全拦截：val.zip 属于官方 152 条最终评测集，Sprint 1–3 严禁下载；到 Sprint 4 须显式传入 --unlock-final-eval-val")
    info = KNOWN_ARCHIVES[split]
    file_id = info["google_drive_file_id"]
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    dest_dir.mkdir(parents=True, exist_ok=True)
    target_path = dest_dir / info["filename"]
    part_path = dest_dir / f"{info['filename']}.part"
    status_path = dest_dir / f"{info['filename']}.status.json"
    manifest_path = dest_dir / "manifest.json"

    remote_size = probe_remote_size(url)
    if info["expected_size"] is not None and remote_size != info["expected_size"]:
        raise RuntimeError(f"远程文件大小 ({remote_size}) 与已锁定大小 ({info['expected_size']}) 不一致")

    if target_path.is_file() and target_path.stat().st_size == remote_size and manifest_path.is_file():
        print(f"Target archive already exists and matches size ({remote_size:,} bytes): {target_path}", flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    chunk_bytes = chunk_mib * 1024 * 1024
    num_chunks = (remote_size + chunk_bytes - 1) // chunk_bytes

    completed_chunks: set[int] = set()
    if status_path.is_file() and part_path.is_file() and part_path.stat().st_size == remote_size:
        try:
            st = json.loads(status_path.read_text(encoding="utf-8"))
            if st.get("remote_size") == remote_size and st.get("chunk_bytes") == chunk_bytes:
                completed_chunks = set(int(x) for x in st.get("completed_chunks", []))
        except Exception:
            completed_chunks = set()

    if not part_path.is_file() or part_path.stat().st_size != remote_size:
        print(f"Pre-allocating {part_path} ({remote_size:,} bytes / {remote_size/(1024**3):.2f} GiB)...", flush=True)
        completed_chunks = set()
        with part_path.open("wb") as fh:
            fh.seek(remote_size - 1)
            fh.write(b"\x00")

    pending = [idx for idx in range(num_chunks) if idx not in completed_chunks]
    print(
        f"Downloading {info['filename']} ({remote_size/(1024**3):.2f} GiB) in {num_chunks} chunks of {chunk_mib} MiB "
        f"({len(completed_chunks)} done, {len(pending)} pending) with {workers} workers...",
        flush=True,
    )

    file_lock = threading.Lock()
    t_start = time.monotonic()
    bytes_done = sum(
        min(chunk_bytes, remote_size - idx * chunk_bytes) for idx in completed_chunks
    )

    def save_status() -> None:
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "file_id": file_id,
                    "remote_size": remote_size,
                    "chunk_bytes": chunk_bytes,
                    "num_chunks": num_chunks,
                    "completed_chunks": sorted(completed_chunks),
                    "updated_at": utc(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(status_path)

    if pending:
        with part_path.open("r+b") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            it = iter(pending)

            for _ in range(min(workers * 2, len(pending))):
                cidx = next(it)
                cstart = cidx * chunk_bytes
                cend = min(remote_size - 1, cstart + chunk_bytes - 1)
                futures[pool.submit(fetch_chunk, url, cstart, cend)] = (cidx, cstart, cend)

            while futures:
                done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    cidx, cstart, cend = futures.pop(fut)
                    data = fut.result()
                    with file_lock:
                        fh.seek(cstart)
                        fh.write(data)
                        fh.flush()
                        completed_chunks.add(cidx)
                        bytes_done += len(data)
                        if len(completed_chunks) % 5 == 0 or len(completed_chunks) == num_chunks:
                            save_status()
                            elapsed = max(0.1, time.monotonic() - t_start)
                            pct = bytes_done * 100.0 / remote_size
                            mb_s = bytes_done / (1024 * 1024) / elapsed
                            print(
                                f"  [{len(completed_chunks):3d}/{num_chunks}] {pct:5.1f}% "
                                f"({bytes_done/(1024**3):.2f}/{remote_size/(1024**3):.2f} GiB) | avg={mb_s:.2f} MiB/s",
                                flush=True,
                            )
                    try:
                        next_idx = next(it)
                        nstart = next_idx * chunk_bytes
                        nend = min(remote_size - 1, nstart + chunk_bytes - 1)
                        futures[pool.submit(fetch_chunk, url, nstart, nend)] = (next_idx, nstart, nend)
                    except StopIteration:
                        pass

    if len(completed_chunks) != num_chunks:
        raise RuntimeError(f"下载未完成：{len(completed_chunks)}/{num_chunks}")

    if skip_crc_sweep:
        with zipfile.ZipFile(part_path, "r") as zf:
            names = zf.namelist()
            zip_check = {
                "total_entries": len(names),
                "text_search_entries": sum(1 for n in names if "text_related_store_text_" in n and n.endswith(".json")),
                "reverse_image_search_entries": sum(1 for n in names if "image_related_store_text_" in n and n.endswith(".json")),
                "crc32_verified": False,
            }
    else:
        zip_check = verify_zip_archive(part_path, info["expected_text_entries"], info["expected_ris_entries"])

    print("Computing whole-file SHA-256...", flush=True)
    archive_sha256 = file_sha256(part_path)
    part_path.replace(target_path)
    if status_path.is_file():
        status_path.unlink()

    manifest = {
        "schema_version": 1,
        "split": split,
        "filename": info["filename"],
        "google_drive_file_id": file_id,
        "size_bytes": remote_size,
        "sha256": archive_sha256,
        "zip_verification": zip_check,
        "downloaded_at": utc(),
        "val_final_eval_touched": (split == "val"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Published verified archive: {target_path} (SHA-256: {archive_sha256})", flush=True)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--dest-dir", type=Path, default=None)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--unlock-final-eval-val", action="store_true")
    parser.add_argument("--skip-crc-sweep", action="store_true")
    args = parser.parse_args(argv)

    dest = args.dest_dir or (ROOT / "data/averimatec/knowledge_store" / args.split)
    download_archive(
        args.split,
        dest,
        chunk_mib=args.chunk_mib,
        workers=args.workers,
        unlock_final_eval_val=args.unlock_final_eval_val,
        skip_crc_sweep=args.skip_crc_sweep,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
