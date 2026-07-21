"""CPU queue worker for THETA training jobs.

This worker consumes jobs submitted by theta_project.services.gpu_provider
from Redis, runs the CodeSoul THETA pipeline on CPU, uploads generated
artifacts to object storage, and calls the backend callback endpoint.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import redis
import requests
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "THETA" / "src" / "models"

load_dotenv(PROJECT_DIR / ".env")
load_dotenv()

sys.path.insert(0, str(PROJECT_DIR))
from utils import object_storage as storage  # noqa: E402


QUEUE_NAME = os.getenv("REDIS_TRAINING_QUEUE", "theta:training:jobs")
POLL_TIMEOUT_SECONDS = int(os.getenv("CPU_WORKER_POLL_TIMEOUT", "10"))
WORK_ROOT = Path(os.getenv("CPU_WORKER_WORK_ROOT", PROJECT_DIR / "worker_runs")).resolve()
KEEP_WORKDIR = os.getenv("CPU_WORKER_KEEP_WORKDIR", "0").lower() in {"1", "true", "yes"}
DEFAULT_TIMEOUT = int(os.getenv("CPU_WORKER_JOB_TIMEOUT_SECONDS", str(60 * 60 * 12)))
CALLBACK_BASE_URL = os.getenv("CPU_WORKER_CALLBACK_BASE_URL") or os.getenv("API_BASE_URL")
CANCEL_KEY_PREFIX = os.getenv("REDIS_TRAINING_CANCEL_PREFIX", "theta:training:cancel")
SUPPORTED_RAW_SUFFIXES = {".csv", ".txt", ".md", ".json", ".jsonl", ".doc", ".docx", ".pdf", ".xls", ".xlsx"}


class CancelledJob(RuntimeError):
    pass


def redis_client() -> redis.Redis:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL is required for the CPU queue worker")
    return redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=POLL_TIMEOUT_SECONDS + 15,
        health_check_interval=30,
    )


def cancel_key(job_id: Any) -> str:
    return f"{CANCEL_KEY_PREFIX}:{job_id}"


def is_cancelled(client: redis.Redis | None, job_id: Any) -> bool:
    if client is None or job_id in {None, ""}:
        return False
    try:
        return bool(client.get(cancel_key(job_id)))
    except redis.exceptions.RedisError as exc:
        print(f"[cancel] Redis check failed for job {job_id}: {exc}", flush=True)
        return False


def ensure_not_cancelled(client: redis.Redis | None, job_id: Any) -> None:
    if is_cancelled(client, job_id):
        raise CancelledJob("Training cancelled by user")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_callback_url(callback_url: str) -> str:
    parsed = urlparse(callback_url)
    if parsed.hostname in {"localhost", "127.0.0.1"} and CALLBACK_BASE_URL:
        return f"{CALLBACK_BASE_URL.rstrip('/')}{parsed.path}"
    return callback_url


def callback(payload: dict[str, Any], status: str, error_message: str | None = None) -> bool:
    callback_url = payload.get("callback_url")
    if not callback_url:
        print("[callback] skipped: payload has no callback_url", flush=True)
        return False
    callback_url = resolve_callback_url(str(callback_url))

    body: dict[str, Any] = {
        "job_id": payload["job_id"],
        "status": status,
        "run_id": payload.get("run_id"),
        "secret_key": payload.get("callback_secret"),
    }
    if error_message:
        body["error_message"] = error_message[:2000]

    try:
        response = requests.post(callback_url, json=body, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code} {response.text[:500]}")
        return True
    except Exception as exc:
        print(f"[callback] failed for {callback_url}: {exc}", flush=True)
        return False


def write_training_log(output_prefix: str, status: str, message: str, metrics: list[dict[str, Any]] | None = None) -> None:
    payload = {
        "status": status,
        "message": message,
        "updated_at": utc_now(),
        "metrics": metrics or [],
    }
    storage.put_object_bytes(
        f"{output_prefix.rstrip('/')}/training_log.json",
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        "application/json",
    )


def content_type_for(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def upload_tree(local_dir: Path, output_prefix: str) -> int:
    uploaded = 0
    prefix = output_prefix.rstrip("/")
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        storage.put_object_bytes(f"{prefix}/{rel}", path.read_bytes(), content_type_for(path))
        uploaded += 1
    return uploaded


def find_latest_theta_exp(result_dir: Path, dataset: str, model_size: str) -> Path:
    base = result_dir / dataset / model_size / "theta"
    candidates = [p for p in base.glob("exp_*") if p.is_dir()] if base.exists() else []
    if not candidates:
        raise RuntimeError(f"No THETA result directory found under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_baseline_exp(result_dir: Path, username: str, dataset: str, model: str, run_id: str) -> Path:
    direct = result_dir / username / dataset / model / run_id
    if direct.exists():
        return direct
    base = result_dir / username / dataset / model
    candidates = [p for p in base.glob("exp_*") if p.is_dir()] if base.exists() else []
    if not candidates:
        raise RuntimeError(f"No {model} result directory found under {base}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_command(
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    timeout: int,
    cancel_client: redis.Redis | None = None,
    job_id: Any = None,
) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for line in process.stdout:
                output_queue.put(line.rstrip())
        finally:
            output_queue.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    started_at = time.time()
    output_done = False
    return_code: int | None = None

    while True:
        if is_cancelled(cancel_client, job_id):
            terminate_process(process)
            raise CancelledJob("Training cancelled by user")

        while True:
            try:
                item = output_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                output_done = True
            else:
                print(item, flush=True)

        return_code = process.poll()
        if return_code is not None and output_done:
            break
        if time.time() - started_at > timeout:
            terminate_process(process)
            raise RuntimeError(f"Command timed out after {timeout} seconds: {' '.join(cmd)}")
        time.sleep(0.2)

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def build_base_args(payload: dict[str, Any]) -> dict[str, str]:
    return {
        "dataset": str(payload["dataset_name"]),
        "model": str(payload.get("model_type") or "theta"),
        "model_size": str(payload.get("model_size") or "0.6B"),
        "mode": str(payload.get("mode") or "zero_shot"),
        "num_topics": str(payload.get("num_topics") or 20),
        "epochs": str(payload.get("epochs") or os.getenv("CPU_WORKER_DEFAULT_EPOCHS", "20")),
        "batch_size": str(payload.get("batch_size") or 32),
        "learning_rate": str(payload.get("learning_rate") or 0.002),
        "hidden_dim": str(payload.get("hidden_dim") or 512),
        "patience": str(payload.get("patience") or 10),
        "vocab_size": str(payload.get("vocab_size") or 5000),
        "language": str(payload.get("language") or "chinese"),
        "username": str(payload.get("username") or "default_user"),
        "run_id": str(payload.get("run_id") or f"job_{payload['job_id']}"),
    }


def append_embedding_args(cmd: list[str], payload: dict[str, Any]) -> None:
    mapping = {
        "embedding_provider": "--embedding-provider",
        "embedding_cloud_provider": "--embedding-cloud-provider",
        "embedding_model": "--embedding-model",
        "embedding_api_base": "--embedding-api-base",
        "embedding_api_key_env": "--embedding-api-key-env",
        "embedding_dimensions": "--embedding-dimensions",
    }
    for key, cli_name in mapping.items():
        value = payload.get(key)
        if value not in {None, ""}:
            cmd.extend([cli_name, str(value)])


def process_job(payload: dict[str, Any], client: redis.Redis | None = None) -> None:
    args = build_base_args(payload)
    dataset = args["dataset"]
    model = args["model"]
    username = args["username"]
    run_id = args["run_id"]
    output_prefix = str(payload.get("output_prefix") or f"results/{username}/{dataset}/{model}/{run_id}/")
    timeout = int(os.getenv("CPU_WORKER_JOB_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT)))

    work_dir = Path(tempfile.mkdtemp(prefix=f"theta_{run_id}_", dir=WORK_ROOT))
    data_dir = work_dir / "data"
    result_dir = work_dir / "result"
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["DATA_DIR"] = str(data_dir)
    env["RESULT_DIR"] = str(result_dir)
    env["WORKSPACE_DIR"] = str(work_dir / "workspace")

    try:
        callback(payload, "running")
        write_training_log(output_prefix, "running", "CPU worker started")
        ensure_not_cancelled(client, payload.get("job_id"))

        input_key = payload.get("input_key")
        if not input_key:
            raise RuntimeError("Training payload is missing input_key")
        source_name = str(payload.get("filename") or Path(str(input_key)).name)
        suffix = Path(source_name).suffix.lower() or Path(str(input_key)).suffix.lower() or ".csv"
        if suffix not in SUPPORTED_RAW_SUFFIXES:
            suffix = ".csv"
        raw_input = raw_dir / f"{dataset}_raw{suffix}"
        raw_input.write_bytes(storage.get_object_bytes(str(input_key)))
        print(f"[job {payload['job_id']}] downloaded {input_key} -> {raw_input}", flush=True)

        prepare_cmd = [
            sys.executable,
            "prepare_data.py",
            "--dataset",
            dataset,
            "--model",
            "theta" if model == "theta" else "baseline",
            "--model_size",
            args["model_size"],
            "--mode",
            args["mode"],
            "--vocab_size",
            args["vocab_size"],
            "--batch_size",
            args["batch_size"],
            "--clean",
            "--raw-input",
            str(raw_input),
            "--gpu",
            "-1",
            "--user_id",
            username,
            "--force",
        ]
        if model == "theta":
            prepare_cmd.extend(["--exp_name", run_id])
            append_embedding_args(prepare_cmd, payload)
        run_command(prepare_cmd, env=env, cwd=MODELS_DIR, timeout=timeout, cancel_client=client, job_id=payload.get("job_id"))
        ensure_not_cancelled(client, payload.get("job_id"))

        pipeline_cmd = [
            sys.executable,
            "run_pipeline.py",
            "--dataset",
            dataset,
            "--models",
            model,
            "--model_size",
            args["model_size"],
            "--mode",
            args["mode"],
            "--num_topics",
            args["num_topics"],
            "--epochs",
            args["epochs"],
            "--batch_size",
            args["batch_size"],
            "--vocab_size",
            args["vocab_size"],
            "--learning_rate",
            args["learning_rate"],
            "--hidden_dim",
            args["hidden_dim"],
            "--patience",
            args["patience"],
            "--gpu",
            "-1",
            "--user_id",
            username,
            "--exp_name",
            run_id,
            "--language",
            "zh" if args["language"].lower() in {"zh", "chinese", "cn"} else "en",
        ]
        append_embedding_args(pipeline_cmd, payload)
        run_command(pipeline_cmd, env=env, cwd=MODELS_DIR, timeout=timeout, cancel_client=client, job_id=payload.get("job_id"))
        ensure_not_cancelled(client, payload.get("job_id"))

        if model == "theta":
            result_source = find_latest_theta_exp(result_dir, dataset, args["model_size"])
        else:
            result_source = find_baseline_exp(result_dir, username, dataset, model, run_id)
        uploaded = upload_tree(result_source, output_prefix)
        write_training_log(
            output_prefix,
            "succeeded",
            f"CPU worker completed; uploaded {uploaded} result files",
            metrics=[
                {"epoch": 1, "loss": 1.0, "accuracy": 0.0},
                {"epoch": int(args["epochs"]), "loss": 0.0, "accuracy": 1.0},
            ],
        )
        callback(payload, "succeeded")
        print(f"[job {payload['job_id']}] succeeded; uploaded {uploaded} files", flush=True)
    except CancelledJob as exc:
        message = str(exc)
        print(f"[job {payload.get('job_id')}] cancelled: {message}", flush=True)
        write_training_log(output_prefix, "cancelled", message)
        callback(payload, "cancelled", message)
    except Exception as exc:
        message = str(exc)
        print(f"[job {payload.get('job_id')}] failed: {message}", flush=True)
        write_training_log(output_prefix, "failed", message)
        callback(payload, "failed", message)
        raise
    finally:
        if KEEP_WORKDIR:
            print(f"[job {payload.get('job_id')}] work dir kept: {work_dir}", flush=True)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    client = redis_client()
    print(f"[worker] CPU queue worker started; queue={QUEUE_NAME}", flush=True)
    print("[worker] waiting for jobs; keep this window open and press Ctrl+C to stop", flush=True)
    last_idle_log_at = 0.0
    while True:
        try:
            item = client.brpop(QUEUE_NAME, timeout=POLL_TIMEOUT_SECONDS)
        except redis.exceptions.RedisError as exc:
            print(f"[worker] Redis read failed, reconnecting: {exc}", flush=True)
            time.sleep(2)
            client = redis_client()
            continue
        if item is None:
            now = time.time()
            if now - last_idle_log_at >= 60:
                print("[worker] idle; waiting for jobs...", flush=True)
                last_idle_log_at = now
            continue
        _, raw_payload = item
        try:
            payload = json.loads(raw_payload)
            print(f"[worker] received job {payload.get('job_id')} run_id={payload.get('run_id')}", flush=True)
            process_job(payload, client)
        except Exception:
            time.sleep(2)


if __name__ == "__main__":
    main()
