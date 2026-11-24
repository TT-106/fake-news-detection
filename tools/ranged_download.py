"""从锁定的 Hugging Face revision 严格分块下载，整文件校验后才发布模型权重。"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from urllib.parse import quote, urlencode
from uuid import uuid4

import requests

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    target.relative_to(root)
    return target


class DownloadError(RuntimeError):
    pass


class RangedDownloader:
    def __init__(self, manifest, destination, *, chunk_bytes=32 * 1024 * 1024, workers=4,
                 connect_timeout=15, read_timeout=30, chunk_deadline=150, max_seconds=3600,
                 attempts=3, ignore_prefix=False, endpoint="https://huggingface.co"):
        self.manifest_path = Path(manifest).resolve()
        self.destination = Path(destination).resolve()
        self.destination.relative_to(ROOT.resolve())
        self.lock = load(self.manifest_path)
        if self.lock.get("repo_type") != "model" or not re.fullmatch(r"[a-f0-9]{40}", self.lock.get("revision", "")):
            raise DownloadError("需要已锁定40位revision的model manifest")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.lock.get("repo_id", "")):
            raise DownloadError("无效repo_id")
        if not 1 <= workers <= 8 or not 1024 <= chunk_bytes <= 256 * 1024 * 1024:
            raise DownloadError("workers须1—8，chunk大小须1KiB—256MiB")
        self.entries = self.lock["files"]
        if endpoint.rstrip("/") not in {"https://huggingface.co", "https://hf-mirror.com", "https://modelscope.cn"}:
            raise DownloadError("本工具仅允许已知HF或ModelScope传输；最终真实性仍以官方锁定SHA为准")
        self.endpoint = endpoint.rstrip("/")
        for entry in self.entries:
            inside(self.destination, entry["path"])
            if type(entry.get("size")) is not int or entry["size"] <= 0:
                raise DownloadError("文件必须有正整数锁定size")
            if entry.get("lfs_sha256") and not re.fullmatch(r"[a-f0-9]{64}", entry["lfs_sha256"]):
                raise DownloadError("无效LFS SHA-256")
        self.chunk_bytes, self.workers = chunk_bytes, workers
        self.connect_timeout, self.read_timeout = connect_timeout, read_timeout
        self.chunk_deadline, self.max_seconds = chunk_deadline, max_seconds
        self.attempts, self.ignore_prefix = attempts, ignore_prefix
        self.cache = ROOT / ".cache/model-downloads" / self.lock["repo_id"].replace("/", "--") / self.lock["revision"]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.cancel = threading.Event()
        self.mutex = threading.Lock()
        self.inflight = {}
        self.completed_bytes = 0
        self.started = time.monotonic()
        self.total_bytes = sum(e["size"] for e in self.entries)
        self.verified_files = []
        previous_path = self.cache / "status.json"
        try:
            previous = load(previous_path) if previous_path.is_file() else {}
        except (OSError, ValueError):
            previous = {}
        self.known_small = {entry["path"]: entry for entry in previous.get("verified_files", [])
                            if previous.get("revision") == self.lock["revision"] and entry.get("verification") == "pinned_revision_git_blob_sha1"}
        self.transport_revisions = {}
        if self.endpoint == "https://modelscope.cn":
            self.setup_modelscope_transport()

    def setup_modelscope_transport(self):
        url = "https://modelscope.cn/api/v1/models/" + self.lock["repo_id"] + "/repo/files"
        for attempt in range(1, self.attempts+1):
            try:
                with self.session() as session, session.get(url, params={"Revision": "master", "Recursive": "true"},
                    timeout=(self.connect_timeout, self.read_timeout)) as response:
                    if response.status_code != 200:
                        raise DownloadError(f"ModelScope元数据HTTP{response.status_code}")
                    data = response.json()
                listing = {entry["Path"]: entry for entry in data["Data"]["Files"]}
                validated = []
                for entry in self.entries:
                    if not entry.get("lfs_sha256"):
                        continue
                    mirror = listing[entry["path"]]
                    if mirror.get("Sha256") != entry["lfs_sha256"] or mirror.get("Size") != entry["size"]:
                        raise DownloadError(f"ModelScope {entry['path']} 与官方锁定SHA/size不同，拒绝切换")
                    if not re.fullmatch(r"[a-f0-9]{40}", mirror.get("Revision", "")):
                        raise DownloadError("ModelScope未提供可锁定文件revision")
                    self.transport_revisions[entry["path"]] = mirror["Revision"]
                    validated.append({"path": entry["path"], "modelscope_revision": mirror["Revision"], "sha256": mirror["Sha256"], "size": mirror["Size"]})
                atomic_json(self.cache / "modelscope_transport_lock.json", {"official_hf_revision": self.lock["revision"],
                    "transport_repo_id": self.lock["repo_id"], "transport_provider": "modelscope.cn", "files": validated,
                    "note": "ModelScope仅提供相同字节传输；最终每文件仍按原始官方HF LFS SHA-256验真。"})
                self.emit("modelscope_metadata_matches_official_lfs", files=len(validated))
                return
            except (requests.RequestException, DownloadError, ValueError, KeyError) as exc:
                self.emit("modelscope_metadata_attempt_failed", attempt=attempt,
                          reason=str(exc) if isinstance(exc, DownloadError) else type(exc).__name__)
                if attempt == self.attempts:
                    raise DownloadError("ModelScope传输元数据有限校验失败") from None
                if self.cancel.wait(min(2*attempt, 5)):
                    raise DownloadError("下载已取消") from None

    def source_url(self, entry):
        endpoint = self.endpoint if entry.get("lfs_sha256") else "https://huggingface.co"
        if endpoint == "https://modelscope.cn":
            return endpoint + "/api/v1/models/" + self.lock["repo_id"] + "/repo?" + urlencode({
                "Revision": self.transport_revisions[entry["path"]], "FilePath": entry["path"]})
        return endpoint + "/" + self.lock["repo_id"] + "/resolve/" + self.lock["revision"] + "/" + quote(entry["path"], safe="/")

    def directory(self, entry):
        target = self.cache / (hashlib.sha256(entry["path"].encode()).hexdigest()[:12] + "-" + Path(entry["path"]).name)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def emit(self, event, **fields):
        payload = {"time": utc(), "event": event, **fields}
        with self.mutex:
            with (self.cache / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def check_cancel(self):
        if self.cancel.is_set():
            raise DownloadError("下载已取消")
        if time.monotonic() - self.started > self.max_seconds:
            self.cancel.set()
            raise DownloadError("本轮下载达到有限总时限；已完成分块已保存，可重新执行续传")

    def status(self, state="running", **fields):
        with self.mutex:
            inflight = sum(self.inflight.values())
            complete = self.completed_bytes
        payload = {"time": utc(), "status": state, "repo_id": self.lock["repo_id"], "revision": self.lock["revision"],
                   "completed_bytes": complete, "inflight_bytes": inflight, "total_bytes": self.total_bytes,
                   "verified_files": self.verified_files, "elapsed_seconds": round(time.monotonic()-self.started, 1), **fields}
        atomic_json(self.cache / "status.json", payload)
        return payload

    def session(self):
        session = requests.Session()
        session.max_redirects = 5
        session.headers.update({"Accept-Encoding": "identity", "User-Agent": "locked-research-range-downloader/1"})
        return session

    def fetch_range(self, entry, start, end, target):
        target = Path(target)
        expected = end-start+1
        identity = entry["path"] + f":{start}-{end}"
        for attempt in range(1, self.attempts+1):
            self.check_cancel()
            tmp = target.with_name(target.name + "." + uuid4().hex + ".part")
            try:
                tick = time.monotonic()
                with self.session() as session, session.get(self.source_url(entry), headers={"Range": f"bytes={start}-{end}"},
                            stream=True, timeout=(self.connect_timeout, self.read_timeout)) as response:
                    # 不打印 response.url 或异常正文，重定向URL可能携带短期签名。
                    if response.status_code != 206:
                        raise DownloadError(f"range HTTP状态须206，实际{response.status_code}")
                    expected_range = f"bytes {start}-{end}/{entry['size']}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise DownloadError("Content-Range与请求/锁定size不一致")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) != expected:
                        raise DownloadError("Content-Length与范围不一致")
                    if response.headers.get("Content-Encoding", "identity").lower() not in {"identity", ""}:
                        raise DownloadError("拒绝改变字节语义的Content-Encoding")
                    size, sha = 0, hashlib.sha256()
                    with tmp.open("xb") as stream:
                        for block in response.iter_content(1024*1024):
                            self.check_cancel()
                            if time.monotonic()-tick > self.chunk_deadline:
                                raise DownloadError("单块达到有限时限")
                            if not block:
                                continue
                            size += len(block)
                            if size > expected:
                                raise DownloadError("响应字节超过请求范围")
                            stream.write(block)
                            sha.update(block)
                            with self.mutex:
                                self.inflight[identity] = size
                        stream.flush()
                        os.fsync(stream.fileno())
                    if size != expected:
                        raise DownloadError("响应提前结束，未达到范围长度")
                    tmp.replace(target)
                    result = {"start": start, "end": end, "size": size, "sha256": sha.hexdigest(), "transport_endpoint": self.endpoint,
                              "source": "strict_http_206", "revision": self.lock["revision"], "lfs_sha256": entry.get("lfs_sha256")}
                    atomic_json(target.with_suffix(".json"), result)
                    self.emit("chunk_completed", file=entry["path"], start=start, end=end, bytes=size, attempt=attempt,
                              elapsed_seconds=round(time.monotonic()-tick, 1))
                    return result
            except (requests.RequestException, DownloadError, OSError, ValueError) as exc:
                tmp.unlink(missing_ok=True)
                # 仅暴露受控错误，不输出 requests 的带签名URL异常字符串。
                detail = str(exc) if isinstance(exc, DownloadError) else type(exc).__name__
                self.emit("chunk_attempt_failed", file=entry["path"], start=start, end=end, attempt=attempt, reason=detail)
                if attempt == self.attempts:
                    raise DownloadError(f"{entry['path']} bytes={start}-{end} 有限重试耗尽") from None
                if self.cancel.wait(min(2*attempt, 5)):
                    raise DownloadError("下载已取消") from None
            finally:
                with self.mutex:
                    self.inflight.pop(identity, None)

    def preserve_prefixes(self, entry):
        directory = self.directory(entry)
        if not entry.get("lfs_sha256"):
            return []
        original_cache = self.destination / ".cache/huggingface/download"
        for source in original_cache.glob("*." + entry["lfs_sha256"] + ".incomplete"):
            source = source.resolve()
            source.relative_to(self.destination.resolve())
            size = source.stat().st_size
            if not 0 < size <= entry["size"]:
                continue
            target = directory / f"hf-prefix-{size}-{uuid4().hex[:8]}.bin"
            # 同盘原子移动保留前缀，既不重读网络，也不删除未核验数据。
            source.replace(target)
            self.emit("hf_prefix_preserved", file=entry["path"], bytes=size, cache_file=target.relative_to(ROOT).as_posix())
        return sorted(directory.glob("hf-prefix-*.bin"), key=lambda p: p.stat().st_size, reverse=True)

    def cached_chunk(self, entry, start, end, target):
        metadata = target.with_suffix(".json")
        if not target.is_file() or not metadata.is_file():
            return False
        try:
            data = load(metadata)
            if data["start"] != start or data["end"] != end or data["revision"] != self.lock["revision"] or data["lfs_sha256"] != entry["lfs_sha256"]:
                return False
            if self.ignore_prefix and data["source"] == "unverified_hf_prefix":
                return False
            return target.stat().st_size == end-start+1 and file_sha(target) == data["sha256"]
        except (OSError, ValueError, KeyError):
            return False

    def seed_from_prefix(self, entry, start, end, target, prefix):
        tmp = target.with_name(target.name + "." + uuid4().hex + ".part")
        expected = end-start+1
        with prefix.open("rb") as stream:
            stream.seek(start)
            raw = stream.read(expected)
        if len(raw) != expected:
            raise DownloadError("保留的连续前缀长度不一致")
        tmp.write_bytes(raw)
        tmp.replace(target)
        atomic_json(target.with_suffix(".json"), {"start": start, "end": end, "size": expected,
            "sha256": hashlib.sha256(raw).hexdigest(), "source": "unverified_hf_prefix",
            "revision": self.lock["revision"], "lfs_sha256": entry["lfs_sha256"]})

    def download_small(self, entry):
        target = inside(self.destination, entry["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        cached = self.known_small.get(entry["path"])
        if cached and target.is_file() and target.stat().st_size == entry["size"] and file_sha(target) == cached.get("sha256"):
            blob = hashlib.sha1(f"blob {entry['size']}\0".encode())
            with target.open("rb") as stream:
                while block := stream.read(1024*1024):
                    blob.update(block)
            if blob.hexdigest() == cached.get("git_blob_sha1"):
                with self.mutex:
                    self.completed_bytes += entry["size"]
                self.emit("small_file_cached_hash_verified", file=entry["path"], bytes=entry["size"])
                return dict(entry, sha256=cached["sha256"], verified_size=entry["size"],
                            git_blob_sha1=cached["git_blob_sha1"], verification="pinned_revision_git_blob_sha1")
        # Git存储的小文件按锁定revision读取，ETag是Git blob SHA-1，不当作SHA-256。
        for attempt in range(1, self.attempts+1):
            self.check_cancel()
            tmp = self.directory(entry) / (uuid4().hex + ".small.part")
            try:
                tick = time.monotonic()
                with self.session() as session, session.get(self.source_url(entry), stream=True,
                    timeout=(self.connect_timeout, self.read_timeout)) as response:
                    if response.status_code != 200:
                        raise DownloadError(f"small HTTP状态{response.status_code}")
                    responses = [*response.history, response]
                    commits = {r.headers.get("X-Repo-Commit") for r in responses if r.headers.get("X-Repo-Commit")}
                    if self.lock["revision"] not in commits or len(commits) != 1:
                        raise DownloadError("小文件响应未证实锁定revision")
                    etags = [(r.headers.get("ETag") or "").strip('"') for r in responses]
                    blob = next((v for v in reversed(etags) if re.fullmatch(r"[a-f0-9]{40}", v)), None)
                    if blob is None:
                        raise DownloadError("小文件缺少可验证Git blob ETag")
                    size = 0
                    git_sha = hashlib.sha1(f"blob {entry['size']}\0".encode())
                    with tmp.open("xb") as stream:
                        for block in response.iter_content(1024*1024):
                            self.check_cancel()
                            if time.monotonic()-tick > self.chunk_deadline:
                                raise DownloadError("小文件达到有限时限")
                            size += len(block)
                            if size > entry["size"]:
                                raise DownloadError("小文件超过锁定size")
                            stream.write(block)
                            git_sha.update(block)
                    if size != entry["size"] or git_sha.hexdigest() != blob:
                        raise DownloadError("小文件size或Git blob SHA不匹配")
                    actual = file_sha(tmp)
                    tmp.replace(target)
                    with self.mutex:
                        self.completed_bytes += size
                    self.emit("small_file_verified", file=entry["path"], bytes=size, git_blob_sha1=blob)
                    return dict(entry, sha256=actual, verified_size=size, git_blob_sha1=blob, verification="pinned_revision_git_blob_sha1")
            except (requests.RequestException, DownloadError, OSError, ValueError) as exc:
                tmp.unlink(missing_ok=True)
                self.emit("small_attempt_failed", file=entry["path"], attempt=attempt,
                          reason=str(exc) if isinstance(exc, DownloadError) else type(exc).__name__)
                if attempt == self.attempts:
                    raise DownloadError(f"小文件 {entry['path']} 有限重试耗尽") from None
                if self.cancel.wait(min(2*attempt, 5)):
                    raise DownloadError("下载已取消") from None

    def probe(self, length=1024*1024, offset=0):
        entry = next(e for e in self.entries if e.get("lfs_sha256"))
        end = min(entry["size"]-1, offset+length-1)
        if not 0 <= offset <= end:
            raise DownloadError("probe范围越界")
        result = self.fetch_range(entry, offset, end, self.directory(entry) / "probe.bin")
        atomic_json(self.cache / "probe.json", {"file": entry["path"], "revision": self.lock["revision"], **result})
        self.emit("probe_verified", file=entry["path"], bytes=result["size"], sha256=result["sha256"], transport_endpoint=self.endpoint)

    def run(self):
        tasks, chunks = [], {}
        self.emit("download_started", repo_id=self.lock["repo_id"], revision=self.lock["revision"], chunk_bytes=self.chunk_bytes, workers=self.workers,
                  weight_transport_endpoint=self.endpoint, authority="official_locked_revision_and_lfs_sha256")
        for entry in self.entries:
            if not entry.get("lfs_sha256"):
                continue
            destination = inside(self.destination, entry["path"])
            if destination.is_file() and destination.stat().st_size == entry["size"] and file_sha(destination) == entry["lfs_sha256"]:
                self.verified_files.append(dict(entry, sha256=entry["lfs_sha256"], verified_size=entry["size"], verification="locked_lfs_sha256"))
                self.completed_bytes += entry["size"]
                continue
            directory = self.directory(entry)
            prefixes = self.preserve_prefixes(entry)
            prefix = prefixes[0] if prefixes and not self.ignore_prefix else None
            pieces = []
            seeded = 0
            for start in range(0, entry["size"], self.chunk_bytes):
                self.check_cancel()
                end = min(entry["size"]-1, start+self.chunk_bytes-1)
                target = directory / f"chunk-{start:012d}-{end:012d}.bin"
                pieces.append(target)
                if not self.cached_chunk(entry, start, end, target):
                    if prefix is not None and end < prefix.stat().st_size:
                        self.seed_from_prefix(entry, start, end, target, prefix)
                        seeded += end-start+1
                    else:
                        tasks.append((entry, start, end, target))
                        continue
                self.completed_bytes += end-start+1
            if seeded:
                self.emit("prefix_chunks_seeded", file=entry["path"], bytes=seeded, authenticity="pending_full_lfs_sha256")
            chunks[entry["path"]] = (entry, pieces)
        self.status()
        executor = ThreadPoolExecutor(max_workers=self.workers)
        futures = {}
        try:
            # 小文件也必须来自同一锁定版本并验证，不使用size相等冒充内容相同。
            for entry in self.entries:
                if not entry.get("lfs_sha256"):
                    self.verified_files.append(self.download_small(entry))
            for entry, start, end, target in tasks:
                future = executor.submit(self.fetch_range, entry, start, end, target)
                futures[future] = end-start+1
            pending = set(futures)
            while pending:
                self.check_cancel()
                done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    with self.mutex:
                        self.completed_bytes += futures[future]
                progress = self.status()
                self.emit("progress", completed_bytes=progress["completed_bytes"], inflight_bytes=progress["inflight_bytes"], total_bytes=self.total_bytes,
                          remaining_chunks=len(pending), elapsed_seconds=progress["elapsed_seconds"])
            executor.shutdown(wait=True)
            for entry, pieces in chunks.values():
                self.check_cancel()
                directory = self.directory(entry)
                assembled = directory / ("assembled-" + uuid4().hex + ".tmp")
                size, sha = 0, hashlib.sha256()
                with assembled.open("xb") as out:
                    for piece in pieces:
                        with piece.open("rb") as stream:
                            while block := stream.read(8*1024*1024):
                                self.check_cancel()
                                out.write(block)
                                size += len(block)
                                sha.update(block)
                    out.flush()
                    os.fsync(out.fileno())
                if size != entry["size"] or sha.hexdigest() != entry["lfs_sha256"]:
                    self.emit("full_lfs_hash_mismatch", file=entry["path"], bytes=size, note="缓存保留；可用--ignore-prefix重跑前缀来源块")
                    raise DownloadError("合并文件LFS SHA-256不匹配，未落地模型目录")
                target = inside(self.destination, entry["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                assembled.replace(target)
                verified = dict(entry, sha256=sha.hexdigest(), verified_size=size, verification="locked_lfs_sha256")
                self.verified_files.append(verified)
                self.emit("weight_lfs_verified_and_published", file=entry["path"], bytes=size, sha256=sha.hexdigest())
            final = dict(self.lock, files=sorted(self.verified_files, key=lambda entry: entry["path"]),
                         status="HASH_VERIFIED", finished_at=utc(), download_protocol="STRICT_HTTP_RANGE_WITH_FULL_LFS_SHA256",
                         weight_transport_endpoint=self.endpoint, weight_transport_is_original_hf=self.endpoint == "https://huggingface.co",
                         alternate_transport_file_revisions=self.transport_revisions)
            atomic_json(self.manifest_path, final)
            self.status("HASH_VERIFIED")
            self.emit("all_files_hash_verified", count=len(self.verified_files), total_bytes=self.total_bytes)
        except BaseException as exc:
            self.cancel.set()
            executor.shutdown(wait=True, cancel_futures=True)
            detail = str(exc) if isinstance(exc, DownloadError) else type(exc).__name__
            self.status("FAILED_OR_INTERRUPTED_RESUMABLE", error=detail)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-bytes", type=int, default=1024*1024)
    parser.add_argument("--probe-offset", type=int, default=0)
    parser.add_argument("--ignore-prefix", action="store_true")
    parser.add_argument("--endpoint", default="https://huggingface.co", choices=["https://huggingface.co", "https://hf-mirror.com", "https://modelscope.cn"])
    args = parser.parse_args(argv)
    try:
        downloader = RangedDownloader(args.manifest, args.destination, chunk_bytes=args.chunk_mib*1024*1024,
                                      workers=args.workers, max_seconds=args.max_seconds, ignore_prefix=args.ignore_prefix, endpoint=args.endpoint)
        if args.probe:
            downloader.probe(args.probe_bytes, args.probe_offset)
        else:
            downloader.run()
        return 0
    except (DownloadError, OSError, ValueError, KeyError, requests.RequestException) as exc:
        print("分块下载停止：" + (str(exc) if isinstance(exc, DownloadError) else type(exc).__name__), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("分块下载已中断，已完成块保留可恢复。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
