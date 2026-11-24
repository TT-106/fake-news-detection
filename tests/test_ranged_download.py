"""严格Range下载的字节边界与发布契约；不访问网络。"""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("range_download_test_module", Path(__file__).resolve().parents[1] / "tools/ranged_download.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class Response:
    def __init__(self, body=b"hello", status=206, content_range="bytes 0-4/5", content_length="5"):
        self.body = body
        self.status_code = status
        self.headers = {"Content-Range": content_range, "Content-Length": content_length}
        self.history = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def iter_content(self, size):
        yield self.body


class Session:
    def __init__(self, response, calls):
        self.response, self.calls = response, calls
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "ROOT", tmp_path)
    model = tmp_path / "models/test-model"
    model.mkdir(parents=True)
    entry = {"path": "model.safetensors", "size": 5, "lfs_sha256": hashlib.sha256(b"hello").hexdigest()}
    manifest = model / "UPSTREAM_LOCK.json"
    manifest.write_text(json.dumps({"repo_id": "Test/Model", "repo_type": "model", "revision": "a"*40, "files": [entry]}), encoding="utf-8")
    job = download.RangedDownloader(manifest, model, chunk_bytes=1024, attempts=3)
    monkeypatch.setattr(job.cancel, "wait", lambda timeout: False)
    return job, entry


def test_valid_range_is_atomic_and_resumable_with_chunk_checksum(job, monkeypatch):
    downloader, entry = job
    calls = []
    monkeypatch.setattr(downloader, "session", lambda: Session(Response(), calls))
    target = downloader.directory(entry) / "chunk.bin"
    result = downloader.fetch_range(entry, 0, 4, target)
    assert result["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert downloader.cached_chunk(entry, 0, 4, target)
    assert calls[0][1]["headers"]["Range"] == "bytes=0-4"
    assert len(calls) == 1 and not list(target.parent.glob("*.part"))
    target.write_bytes(b"other")
    assert not downloader.cached_chunk(entry, 0, 4, target)


@pytest.mark.parametrize("response", [Response(status=200), Response(content_range="bytes 1-5/6"),
    Response(content_length="99"), Response(body=b"hey"), Response(body=b"too-many-bytes")])
def test_ignored_or_malformed_range_exhausts_only_finite_retries_without_publish(job, monkeypatch, response):
    downloader, entry = job
    calls = []
    monkeypatch.setattr(downloader, "session", lambda: Session(response, calls))
    target = downloader.directory(entry) / "rejected.bin"
    with pytest.raises(download.DownloadError, match="重试耗尽"):
        downloader.fetch_range(entry, 0, 4, target)
    assert len(calls) == 3
    assert not target.exists() and not list(target.parent.glob("*.part"))


def test_retained_hf_prefix_cannot_publish_without_full_lfs_sha(job):
    downloader, entry = job
    original = downloader.destination / ".cache/huggingface/download" / ("file." + entry["lfs_sha256"] + ".incomplete")
    original.parent.mkdir(parents=True)
    original.write_bytes(b"WRONG")
    with pytest.raises(download.DownloadError, match="LFS SHA-256不匹配"):
        downloader.run()
    assert not (downloader.destination / entry["path"]).exists()
    assert any(p.read_bytes() == b"WRONG" for p in downloader.directory(entry).glob("hf-prefix-*.bin"))
    assert download.load(downloader.manifest_path).get("status") != "HASH_VERIFIED"


def test_valid_retained_hf_prefix_is_verified_before_model_publication(job):
    downloader, entry = job
    original = downloader.destination / ".cache/huggingface/download" / ("file." + entry["lfs_sha256"] + ".incomplete")
    original.parent.mkdir(parents=True)
    original.write_bytes(b"hello")
    downloader.run()
    assert (downloader.destination / entry["path"]).read_bytes() == b"hello"
    lock = download.load(downloader.manifest_path)
    assert lock["revision"] == "a"*40 and lock["status"] == "HASH_VERIFIED"


def test_small_file_checks_git_blob_digest_and_locked_revision(job, monkeypatch):
    downloader, _ = job
    entry = {"path": "config.json", "size": 5, "lfs_sha256": None}
    response = Response(status=200)
    response.headers.update({"X-Repo-Commit": "a"*40, "ETag": hashlib.sha1(b"blob 5\0hello").hexdigest()})
    monkeypatch.setattr(downloader, "session", lambda: Session(response, []))
    actual = downloader.download_small(entry)
    assert actual["sha256"] == hashlib.sha256(b"hello").hexdigest()
    response.headers["X-Repo-Commit"] = "b"*40
    with pytest.raises(download.DownloadError, match="重试耗尽"):
        downloader.download_small(entry)
