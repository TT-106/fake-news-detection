"""离线研究进度账本：追加实验记录、复核证据并生成可携带 HTML。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.server
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SHANGHAI = timezone(timedelta(hours=8))
PROPORTIONS = {"macro_f1", "accuracy", "coverage", "risk", "selective_risk", "forced_macro_f1", "evidence_score", "averitec_score"}
COSTS = {"cost_usd", "mean_cost_usd", "cost_rmb", "mean_cost_rmb", "latency_seconds"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
HEX = re.compile(r"[a-f0-9]{64}\Z")


def now():
    return datetime.now(SHANGHAI).isoformat(timespec="seconds")


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"JSON 中不允许 {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    else:
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(path)


def inside(root, value, must_exist=True):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("证据路径不能为空")
    root = Path(root).resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"证据路径超出项目：{value}") from exc
    if must_exist and not resolved.is_file():
        raise ValueError(f"证据文件不存在：{value}")
    return resolved


def relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def source_fingerprint(root):
    """这是当前文件内容指纹；不冒充 Git 提交或历史执行时的快照。"""
    root = Path(root)
    paths = set(root.glob("*.py")) | set(root.glob("requirements*.txt"))
    for directory, pattern in (("src", "*.py"), ("tools", "*.py"), ("config", "*.json")):
        paths.update((root / directory).rglob(pattern))
    entries = [{"path": relative(root, p), "sha256": digest(p)} for p in sorted(paths) if p.is_file()]
    return hashlib.sha256(canonical(entries).encode("utf-8")).hexdigest(), entries


def environment_snapshot():
    return {
        "captured_at": now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict(sorted((distribution.metadata["Name"], distribution.version) for distribution in importlib.metadata.distributions() if distribution.metadata.get("Name"))),
        "note": "仅记录当前安装版本；不代表 GPU 或指定模型已完成实际验收。",
    }


def validate_metrics(metrics):
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("完成的实验必须提供非空 metrics")
    for name, value in metrics.items():
        if name not in PROPORTIONS | COSTS:
            raise ValueError(f"未注册指标：{name}；请先在工具中定义指标单位和范围")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"指标 {name} 必须是有限数值")
        if value < 0 or (name in PROPORTIONS and value > 1):
            raise ValueError(f"指标 {name} 越界；比例统一使用 0—1")


def validate_predictions(path, n_samples, declared_format=None):
    path = Path(path)
    format_name = declared_format or ({".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl"}.get(path.suffix.lower()))
    if format_name == "json":
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError("JSON 预测文件必须是数组")
    elif format_name == "jsonl":
        rows = []
        for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"预测中不允许 {value}")))
            except ValueError as exc:
                raise ValueError(f"预测文件第 {index} 行不是规范 JSON") from exc
            rows.append(value)
    else:
        raise ValueError("预测文件仅支持 JSON 数组/JSONL；其它扩展名需显式 prediction_format=json/jsonl")
    if len(rows) != n_samples:
        raise ValueError(f"预测条数 {len(rows)} 与 n_samples={n_samples} 不一致")
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"预测第 {index} 条必须是对象")
        sample_id = row.get("sample_id", row.get("claim_id", row.get("id")))
        if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)) or not str(sample_id).strip():
            raise ValueError(f"预测第 {index} 条需有 sample_id/claim_id/id")
        sample_id = str(sample_id)
        if sample_id in seen:
            raise ValueError(f"预测样本 id 重复：{sample_id}")
        seen.add(sample_id)
        prediction = row.get("prediction")
        if not ((isinstance(prediction, str) and prediction.strip()) or (isinstance(prediction, dict) and prediction)):
            raise ValueError(f"预测第 {index} 条需有非空 prediction 字符串或对象")
    return format_name


def validate_source_archive(run, root):
    if run.get("code_fingerprint_origin") != "execution_start":
        return
    snapshot_path = run.get("artifacts", {}).get("source_snapshot")
    # 运行开始的轻量 start 记录尚未附加此证据；完成记录则必须具备。
    if not snapshot_path and run["status"] == "running":
        return
    if not snapshot_path:
        raise ValueError("包装执行记录缺少源码归档清单")
    entries = read_json(inside(root, snapshot_path))
    if not isinstance(entries, list):
        raise ValueError("源码归档清单必须为数组")
    original = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str) or not HEX.fullmatch(entry["sha256"]):
            raise ValueError("源码归档清单格式无效")
        inside(root, entry.get("path"), must_exist=False)
        archive = inside(root, entry.get("archive_path"))
        if digest(archive) != entry["sha256"]:
            raise ValueError(f"归档源码发生变化：{entry['path']}")
        original.append({"path": entry["path"], "sha256": entry["sha256"]})
    if hashlib.sha256(canonical(original).encode("utf-8")).hexdigest() != run["code_fingerprint"]:
        raise ValueError("归档源码指纹与执行前指纹不一致")


def comparison_key(run):
    fields = ("dataset_id", "dataset_revision", "split_id", "split_sha256", "protocol_id", "model_id", "budget_id", "n_samples")
    contract = {key: run[key] for key in fields}
    contract["experiment_stage"] = run.get("experiment_stage", "formal")
    return hashlib.sha256(canonical(contract).encode("utf-8")).hexdigest()


def validate_run(raw, root=ROOT, require_completed=False):
    if not isinstance(raw, dict):
        raise ValueError("实验记录必须是 JSON 对象")
    run = copy.deepcopy(raw)
    run.setdefault("experiment_stage", "formal")
    if run["experiment_stage"] not in {"formal", "pilot", "engineering"}:
        raise ValueError("experiment_stage 必须为 formal/pilot/engineering")
    for key in ("id", "version", "title", "dataset_id", "dataset_revision", "split_id", "protocol_id", "model_id", "budget_id"):
        if not isinstance(run.get(key), str) or not run[key].strip():
            raise ValueError(f"缺少非空字段：{key}")
    if not IDENTIFIER.fullmatch(run["id"]):
        raise ValueError("id 只允许英文字母、数字、点、下划线、短横线，长度最多 100")
    if run.get("status") not in {"completed", "diagnostic", "failed", "running"}:
        raise ValueError("status 必须为 completed/diagnostic/failed/running")
    if require_completed and run["status"] != "completed":
        raise ValueError("要求 completed 记录")
    if run.get("kind") not in {"method", "baseline", "ablation"}:
        raise ValueError("kind 必须为 method/baseline/ablation")
    for key in ("method_id", "series_id"):
        if key in run and (not isinstance(run[key], str) or not run[key].strip()):
            raise ValueError(f"{key} 存在时必须为非空字符串")
    run.setdefault("series_id", run.get("method_id", "full_method" if run["kind"] == "method" else run["id"]))
    if run.get("review_status") not in {"unreviewed", "reviewed"}:
        raise ValueError("review_status 必须为 unreviewed/reviewed")
    if isinstance(run.get("n_samples"), bool) or not isinstance(run.get("n_samples"), int) or run["n_samples"] <= 0:
        raise ValueError("n_samples 必须为正整数")
    if isinstance(run.get("seed"), bool) or not isinstance(run.get("seed"), int) or run["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    if not isinstance(run.get("command"), list) or not run["command"] or any(not isinstance(x, str) or not x for x in run["command"]):
        raise ValueError("command 必须为非空字符串数组，以避免 shell 拼接歧义")
    for key in ("started_at", "finished_at"):
        if key == "finished_at" and run["status"] == "running":
            continue
        try:
            moment = datetime.fromisoformat(run[key])
            if moment.tzinfo is None:
                raise ValueError("时间须含时区")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{key} 必须为含时区 ISO 时间") from exc
    if run["status"] != "running" and datetime.fromisoformat(run["finished_at"]) < datetime.fromisoformat(run["started_at"]):
        raise ValueError("finished_at 早于 started_at")
    if not isinstance(run.get("split_sha256"), str) or not HEX.fullmatch(run["split_sha256"]):
        raise ValueError("split_sha256 必须为真实 SHA-256")
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts 必须为对象")
    required = {"config", "split"}
    if run["status"] == "completed":
        required |= {"report", "predictions"}
        validate_metrics(run.get("metrics"))
    elif run.get("metrics"):
        validate_metrics(run["metrics"])
    else:
        run["metrics"] = {}
    checksums = {}
    for name in required:
        path = inside(root, artifacts.get(name))
        artifacts[name] = relative(root, path)
        checksums[name] = digest(path)
    # 其它可选证据也必须实际存在且留在项目内。
    for name, value in artifacts.items():
        path = inside(root, value)
        artifacts[name] = relative(root, path)
        checksums[name] = digest(path)
    if checksums["split"] != run["split_sha256"]:
        raise ValueError("split_sha256 与 split 文件不一致")
    config_path = inside(root, run.get("config_path"))
    run["config_path"] = relative(root, config_path)
    if run["config_path"] != artifacts["config"]:
        raise ValueError("config_path 必须与 artifacts.config 指向同一文件")
    if run["status"] == "completed":
        run["prediction_format"] = validate_predictions(inside(root, artifacts["predictions"]), run["n_samples"], run.get("prediction_format"))
        report = read_json(inside(root, artifacts["report"]))
        if not isinstance(report, dict) or report.get("metrics") != run["metrics"]:
            raise ValueError("report 必须含与记录完全一致的 metrics 对象")
        validate_metrics(report["metrics"])
        # 不将任意 report 的另一个 n_samples 静默当成声明中的样本量。
        if type(report.get("n_samples")) is not int or report["n_samples"] != run["n_samples"]:
            raise ValueError("report.n_samples 必须与记录样本量一致")
    if "code_fingerprint" not in run:
        if run["status"] == "completed":
            raise ValueError("完成记录需要 code_fingerprint；可用 fingerprint 命令获取执行前的源码指纹")
        run["code_fingerprint"], _ = source_fingerprint(root)
        run["code_fingerprint_origin"] = "record_time"
    if not isinstance(run["code_fingerprint"], str) or not HEX.fullmatch(run["code_fingerprint"]):
        raise ValueError("code_fingerprint 必须为 SHA-256")
    run.setdefault("code_fingerprint_origin", "declared")
    validate_source_archive(run, root)
    run["schema_version"] = 1
    run["artifact_checksums"] = checksums
    run["comparison_key"] = comparison_key(run)
    return run


def record(raw, root=ROOT):
    if not isinstance(raw, dict):
        raise ValueError("实验记录必须是 JSON 对象")
    if raw.get("review_status") == "reviewed":
        raise ValueError("新入库记录必须 unreviewed；请使用 review 命令追加具名复核事件")
    run = validate_run(raw, root)
    destination = Path(root) / "progress" / "runs" / (run["id"] + ".json")
    write_json(destination, run, exclusive=True)
    return run


def append_event(event, root=ROOT):
    event = dict(event, timestamp=now(), event_id=uuid4().hex)
    write_json(Path(root) / "progress" / "events" / (event["event_id"] + ".json"), event, exclusive=True)
    return event


def withdraw(run_id, reason, root=ROOT):
    if not isinstance(run_id, str) or not IDENTIFIER.fullmatch(run_id):
        raise ValueError("无效 id")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("撤回理由不能为空")
    if not (Path(root) / "progress" / "runs" / (run_id + ".json")).is_file():
        raise ValueError("只能撤回已入库记录")
    return append_event({"type": "withdrawn", "run_id": run_id, "reason": reason.strip()}, root)


def review(run_id, reviewer, note, root=ROOT, reviewer_kind="human"):
    if not isinstance(run_id, str) or not IDENTIFIER.fullmatch(run_id):
        raise ValueError("无效 id")
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(note, str) or not note.strip():
        raise ValueError("复核者和复核记录不能为空")
    if reviewer_kind not in {"human", "agent"}:
        raise ValueError("reviewer_kind 必须为 human/agent")
    path = Path(root) / "progress" / "runs" / (run_id + ".json")
    stored = read_json(path)
    checked = validate_run(stored, root, require_completed=True)
    if checked["artifact_checksums"] != stored["artifact_checksums"]:
        raise ValueError("证据已变化，不能复核通过；请重新运行并使用新 id")
    return append_event({"type": "reviewed", "run_id": run_id, "reviewer": reviewer.strip(),
                         "reviewer_kind": reviewer_kind, "note": note.strip()}, root)


def collect(root=ROOT):
    root = Path(root)
    def optional(name):
        path = root / "outputs" / "progress" / name
        return read_json(path) if path.is_file() else {}
    module_audit = optional("module_audit.json")
    results_audit = optional("results_audit.json")
    events = sorted((read_json(p) for p in (root / "progress" / "events").glob("*.json")), key=lambda x: (x["timestamp"], x["event_id"]))
    revoked = {x["run_id"] for x in events if x.get("type") == "withdrawn"}
    reviews = {}
    for event in events:
        if event.get("type") == "reviewed" and event.get("reviewer_kind", "human") in {"human", "agent"}:
            reviews.setdefault(event["run_id"], []).append(event)
    runs = []
    for path in (root / "progress" / "runs").glob("*.json"):
        run = read_json(path)
        legacy_stage = "experiment_stage" not in run
        issues = []
        try:
            checked = validate_run(run, root)
            if checked["artifact_checksums"] != run.get("artifact_checksums"):
                issues.append("归档后证据文件发生变化")
            if checked["comparison_key"] != run.get("comparison_key"):
                legacy_fields = ("dataset_id", "dataset_revision", "split_id", "split_sha256", "protocol_id", "model_id", "budget_id", "n_samples")
                legacy_key = hashlib.sha256(canonical({key: run[key] for key in legacy_fields}).encode("utf-8")).hexdigest()
                if not (legacy_stage and run.get("comparison_key") == legacy_key):
                    issues.append("比较协议指纹发生变化")
            run["comparison_key"] = checked["comparison_key"]
        except (ValueError, OSError) as exc:
            issues.append(str(exc))
        run["integrity_issues"] = issues
        run.setdefault("experiment_stage", "formal")
        run_reviews = reviews.get(run["id"], [])
        human_reviews = [e for e in run_reviews if e.get("reviewer_kind", "human") == "human"]
        latest_review = (human_reviews or run_reviews or [None])[-1]
        run["review_status"] = "reviewed" if latest_review else "unreviewed"
        run["reviewer_kind"] = latest_review.get("reviewer_kind", "human") if latest_review else None
        run["reviewer"] = latest_review["reviewer"] if latest_review else None
        run["review_note"] = latest_review.get("note") if latest_review else None
        run["withdrawn"] = run["id"] in revoked
        stage_approved = ((run["experiment_stage"] == "formal" and run["reviewer_kind"] == "human")
                          or (run["experiment_stage"] == "pilot" and latest_review is not None))
        run["eligible"] = run["status"] == "completed" and stage_approved and not run["withdrawn"] and not issues
        runs.append(run)
    registered = {x["id"] for x in runs}
    for start in (root / "progress" / "attempts").glob("*/start.json"):
        attempt = read_json(start)
        if attempt["id"] not in registered:
            attempt.setdefault("experiment_stage", "formal")
            attempt.update(eligible=False, withdrawn=False, integrity_issues=[], reviewer_kind=None)
            runs.append(attempt)
    runs.sort(key=lambda x: (x.get("started_at", ""), x["id"]))
    fingerprint, _ = source_fingerprint(root)
    data = {
        "schema_version": 1,
        "project": {"title": "多模态虚假信息检测 · 研究进度", "root": str(root.resolve()), "proposal": module_audit.get("proposal", {"path": "多模态虚假信息检测_大创立项申请书.docx"}), "source_fingerprint": fingerprint, "version_control": "本目录未作为 Git 提交校验；使用源码内容 SHA-256"},
        "audit": module_audit,
        "research": results_audit,
        "runs": runs,
        "events": events,
        "run_template": read_json(root / "progress" / "run.example.json") if (root / "progress" / "run.example.json").is_file() else {},
    }
    data["content_fingerprint"] = hashlib.sha256(canonical(data).encode("utf-8")).hexdigest()
    data["generated_at"] = now()
    return data


def build(root=ROOT):
    root = Path(root)
    data = collect(root)
    template = root / "progress" / "dashboard.template.html"
    if not template.is_file():
        raise ValueError(f"HTML 模板尚不存在：{template}")
    source = template.read_text(encoding="utf-8")
    if source.count("__PROGRESS_DATA_JSON__") != 1:
        raise ValueError("模板必须且只能包含一次 __PROGRESS_DATA_JSON__")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = source.replace("__PROGRESS_DATA_JSON__", payload)
    destination = root / "研究进度.html"
    temporary = destination.with_name(destination.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(html, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    write_json(root / "progress" / "data.json", data)
    return destination


def execute(manifest, command, root=ROOT):
    """运行授权命令并保存日志；只有规范结果文件才能完成入库。"""
    if not isinstance(manifest, dict):
        raise ValueError("运行清单必须是 JSON 对象")
    root = Path(root)
    raw = copy.deepcopy(manifest)
    raw.update(status="running", review_status="unreviewed", started_at=now(), metrics={}, command=command)
    raw.pop("finished_at", None)
    raw["code_fingerprint"], snapshot = source_fingerprint(root)
    raw["code_fingerprint_origin"] = "execution_start"
    # 目标产物此时可以尚未生成；实际 config / split 必须已存在。
    planned = raw.get("artifacts", {}).copy()
    for name in ("report", "predictions"):
        path = inside(root, planned.get(name), must_exist=False)
        if path.exists():
            raise ValueError(f"新运行的 {name} 路径已存在，请使用新产物目录，防止误接收旧结果：{planned[name]}")
    raw["artifacts"] = {key: value for key, value in planned.items() if key not in {"report", "predictions"}}
    raw = validate_run(raw, root)
    raw["execution_input_paths"] = {name: raw["artifacts"][name] for name in ("config", "split")}
    raw["execution_input_checksums"] = {name: digest(inside(root, path)) for name, path in raw["execution_input_paths"].items()}
    if (root / "progress" / "runs" / (raw["id"] + ".json")).exists():
        raise ValueError("该 id 已入库；请为每次实验使用新的 id")
    attempt_dir = root / "progress" / "attempts" / raw["id"]
    attempt_dir.mkdir(parents=True, exist_ok=False)
    raw["execution_config_path"] = raw["config_path"]
    for name in ("config", "split"):
        source = inside(root, raw["artifacts"][name])
        target = attempt_dir / (name + ".snapshot" + source.suffix)
        shutil.copyfile(source, target)
        raw["artifacts"][name] = relative(root, target)
    raw["config_path"] = raw["artifacts"]["config"]
    raw = validate_run(raw, root)
    for entry in snapshot:
        source = inside(root, entry["path"])
        target = attempt_dir / "source" / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != entry["sha256"]:
            raise ValueError(f"保存源码副本时文件变化：{entry['path']}")
        entry["archive_path"] = relative(root, target)
    write_json(attempt_dir / "start.json", raw, exclusive=True)
    write_json(attempt_dir / "source_snapshot.json", snapshot, exclusive=True)
    write_json(attempt_dir / "environment.json", environment_snapshot(), exclusive=True)
    build(root)
    error = None
    exit_code = None
    try:
        with (attempt_dir / "stdout.log").open("wb") as out, (attempt_dir / "stderr.log").open("wb") as err:
            completed = subprocess.run(command, cwd=root, stdout=out, stderr=err, shell=False)
            exit_code = completed.returncode
    except (OSError, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finished = now()
    current_fingerprint, _ = source_fingerprint(root)
    raw.update(finished_at=finished, exit_code=exit_code)
    raw["artifacts"].update({"stdout": relative(root, attempt_dir / "stdout.log"), "stderr": relative(root, attempt_dir / "stderr.log"), "source_snapshot": relative(root, attempt_dir / "source_snapshot.json"), "environment": relative(root, attempt_dir / "environment.json")})
    if exit_code == 0 and not error:
        try:
            if current_fingerprint != raw["code_fingerprint"]:
                raise ValueError("运行期间源码/配置发生变化，不能认定为可复现完成")
            for name, original_path in raw["execution_input_paths"].items():
                if digest(inside(root, original_path)) != raw["execution_input_checksums"][name]:
                    raise ValueError(f"运行期间原始 {name} 文件发生变化，不能挂用执行前快照")
            for name in ("report", "predictions"):
                source = inside(root, planned[name])
                target = attempt_dir / (name + ".snapshot" + source.suffix)
                shutil.copyfile(source, target)
                raw["artifacts"][name] = relative(root, target)
            report = read_json(inside(root, raw["artifacts"]["report"]))
            if not isinstance(report, dict):
                raise ValueError("结果报告须为 JSON 对象")
            raw["metrics"] = report.get("metrics", {})
            raw["status"] = "completed"
            final = record(raw, root)
        except (ValueError, OSError, KeyError) as exc:
            error = str(exc)
    if exit_code != 0 or error:
        raw.update(status="failed", metrics={}, failure_reason=error or f"命令退出码 {exit_code}")
        # 失败轨迹保留已存在的产物，不把缺失文件伪造成证据。
        raw["artifacts"] = {key: value for key, value in raw["artifacts"].items() if inside(root, value, must_exist=False).is_file()}
        raw["artifacts"].update({key: value for key, value in planned.items() if key in {"report", "predictions"} and key not in raw["artifacts"] and inside(root, value, must_exist=False).is_file()})
        final = record(raw, root)
    append_event({"type": "finished", "run_id": raw["id"], "status": final["status"], "exit_code": exit_code}, root)
    build(root)
    return 0 if final["status"] == "completed" else (exit_code or 2)


def serve(root=ROOT, port=8765):
    root = Path(root)
    lock = threading.Lock()
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path not in {"/", "/index.html", "/progress/data.json", "/%E7%A0%94%E7%A9%B6%E8%BF%9B%E5%BA%A6.html"}:
                self.send_error(404)
                return
            try:
                with lock:
                    build(root)
                    target = root / "progress" / "data.json" if path == "/progress/data.json" else root / "研究进度.html"
                    payload = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8" if path.endswith(".json") else "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (OSError, ValueError) as exc:
                self.send_error(500, "Progress build failed")
                print(str(exc), file=sys.stderr)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"本地进度页：http://127.0.0.1:{server.server_port}/ （Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("build", help="重新核验账本并生成可双击 HTML")
    sub.add_parser("fingerprint", help="打印当前源码指纹；请在执行实验前保存")
    add = sub.add_parser("record", help="追加规范结果清单并更新 HTML")
    add.add_argument("--file", type=Path, required=True)
    remove = sub.add_parser("withdraw", help="追加撤回事件，保留原始记录")
    remove.add_argument("--id", required=True)
    remove.add_argument("--reason", required=True)
    approve = sub.add_parser("review", help="追加具名核验事件；正式实验需人工复核，先导允许机器核验")
    approve.add_argument("--id", required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--note", required=True)
    approve.add_argument("--reviewer-kind", choices=("human", "agent"), default="human")
    run = sub.add_parser("run", help="运行一次实验并自动记录成功/失败轨迹")
    run.add_argument("--file", type=Path, required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    server = sub.add_parser("serve", help="在 localhost 查看并读取最新进度")
    server.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.action == "fingerprint":
            print(source_fingerprint(ROOT)[0])
        elif args.action == "record":
            result = record(read_json(args.file), ROOT)
            print(f"已追加 {result['id']}；{build(ROOT)}")
        elif args.action == "withdraw":
            withdraw(args.id, args.reason, ROOT)
            print(f"已追加撤回事件；{build(ROOT)}")
        elif args.action == "review":
            review(args.id, args.reviewer, args.note, ROOT, reviewer_kind=args.reviewer_kind)
            print(f"已追加复核事件；{build(ROOT)}")
        elif args.action == "run":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command:
                raise ValueError("请在 -- 后提供命令与参数")
            return execute(read_json(args.file), command, ROOT)
        elif args.action == "serve":
            if not 0 <= args.port <= 65535:
                raise ValueError("端口必须在 0—65535 范围")
            serve(ROOT, args.port)
        else:
            print(build(ROOT))
        return 0
    except (ValueError, OSError) as exc:
        print(f"进度工具：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
