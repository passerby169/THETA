import argparse
import json
import mimetypes
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

import requests
from dotenv import load_dotenv

from utils import object_storage as storage

load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "THETA" / "src" / "models"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a THETA GPU training job payload")
    parser.add_argument("--payload", help="Path to a JSON payload file")
    parser.add_argument("--redis-loop", action="store_true", help="Consume jobs from REDIS_TRAINING_QUEUE")
    return parser.parse_args()


def _payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload:
        return json.loads(Path(args.payload).read_text(encoding="utf-8"))
    raw = os.getenv("JOB_PAYLOAD")
    if raw:
        return json.loads(raw)
    return json.load(sys.stdin)


def _callback(payload: dict[str, Any], status: str, error: str | None = None) -> None:
    callback_url = payload.get("callback_url")
    if not callback_url:
        return
    body = {
        "job_id": payload["job_id"],
        "run_id": payload.get("run_id"),
        "status": status,
        "secret_key": payload.get("callback_secret"),
        "error_message": error,
    }
    try:
        requests.post(callback_url, json=body, timeout=20)
    except Exception as exc:
        print(f"[callback] failed: {exc}", flush=True)


def _download_input(payload: dict[str, Any]) -> Path:
    dataset = payload["dataset_name"]
    input_key = payload["input_key"]
    filename = Path(input_key).name or f"{dataset}.csv"
    dataset_dir = MODELS_DIR / "data" / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    local_path = dataset_dir / filename
    local_path.write_bytes(storage.get_object_bytes(input_key))

    canonical = dataset_dir / f"{dataset}.csv"
    if local_path != canonical and not canonical.exists():
        canonical.write_bytes(local_path.read_bytes())
    return canonical if canonical.exists() else local_path


def _add_optional(cmd: list[str], payload: dict[str, Any], key: str, option: str) -> None:
    value = payload.get(key)
    if value is not None and value != "":
        cmd.extend([option, str(value)])


def _run_command(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    print(f"[run] {' '.join(cmd)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        process = subprocess.run(
            cmd,
            cwd=str(MODELS_DIR),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(cmd)}")


def _build_common_env(payload: dict[str, Any], work_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["DATA_DIR"] = str(MODELS_DIR / "data")
    env["WORKSPACE_DIR"] = str(work_dir / "workspace")
    env["RESULT_DIR"] = str(work_dir / "result")
    env["CUDA_VISIBLE_DEVICES"] = str(payload.get("gpu", 0))
    for key, env_key in [
        ("embedding_provider", "EMBEDDING_PROVIDER"),
        ("embedding_cloud_provider", "EMBEDDING_CLOUD_PROVIDER"),
        ("embedding_model", "EMBEDDING_MODEL"),
        ("embedding_api_base", "EMBEDDING_API_BASE"),
        ("embedding_api_key_env", "EMBEDDING_API_KEY_ENV"),
        ("embedding_dimensions", "EMBEDDING_DIMENSIONS"),
    ]:
        value = payload.get(key)
        if value is not None and value != "":
            env[env_key] = str(value)
    return env


def _pipeline_commands(payload: dict[str, Any]) -> list[list[str]]:
    dataset = payload["dataset_name"]
    model_type = payload.get("model_type") or "theta"
    model_size = payload.get("model_size") or "0.6B"
    run_id = payload.get("run_id") or f"job_{payload['job_id']}"

    prepare_model = "theta" if model_type == "theta" else "baseline"
    prepare = [
        sys.executable,
        "prepare_data.py",
        "--dataset",
        dataset,
        "--model",
        prepare_model,
        "--model_size",
        model_size,
        "--mode",
        payload.get("mode") or "zero_shot",
        "--vocab_size",
        str(payload.get("vocab_size") or 5000),
        "--batch_size",
        str(payload.get("batch_size") or 64),
        "--gpu",
        str(payload.get("gpu") or 0),
        "--force",
    ]
    train = [
        sys.executable,
        "run_pipeline.py",
        "--dataset",
        dataset,
        "--models",
        model_type,
        "--model_size",
        model_size,
        "--mode",
        payload.get("mode") or "zero_shot",
        "--num_topics",
        str(payload.get("num_topics") or 20),
        "--epochs",
        str(payload.get("epochs") or 100),
        "--batch_size",
        str(payload.get("batch_size") or 64),
        "--learning_rate",
        str(payload.get("learning_rate") or 0.002),
        "--hidden_dim",
        str(payload.get("hidden_dim") or 512),
        "--patience",
        str(payload.get("patience") or 10),
        "--vocab_size",
        str(payload.get("vocab_size") or 5000),
        "--gpu",
        str(payload.get("gpu") or 0),
        "--task_name",
        run_id,
        "--language",
        payload.get("language") or "chinese",
    ]
    for cmd in [prepare, train]:
        _add_optional(cmd, payload, "embedding_provider", "--embedding-provider")
        _add_optional(cmd, payload, "embedding_cloud_provider", "--embedding-cloud-provider")
        _add_optional(cmd, payload, "embedding_model", "--embedding-model")
        _add_optional(cmd, payload, "embedding_api_base", "--embedding-api-base")
        _add_optional(cmd, payload, "embedding_api_key_env", "--embedding-api-key-env")
        _add_optional(cmd, payload, "embedding_dimensions", "--embedding-dimensions")
    return [prepare, train]


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for base in paths:
        if base.exists():
            yield from (p for p in base.rglob("*") if p.is_file())


def _upload_outputs(payload: dict[str, Any], work_dir: Path, log_path: Path) -> int:
    output_prefix = payload["output_prefix"].rstrip("/") + "/"
    result_dir = work_dir / "result"
    roots = [result_dir]
    uploaded = 0
    for file_path in _iter_files(roots):
        relative = file_path.relative_to(result_dir).as_posix()
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        storage.put_object_bytes(output_prefix + relative, file_path.read_bytes(), content_type=content_type)
        uploaded += 1

        if file_path.name in {"topic_words.json", "metrics.json", "training_log.json"}:
            storage.put_object_bytes(output_prefix + file_path.name, file_path.read_bytes(), content_type=content_type)

    storage.put_object_bytes(
        output_prefix + "training_log.txt",
        log_path.read_bytes(),
        content_type="text/plain; charset=utf-8",
    )
    return uploaded


def run_job(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = payload.get("run_id") or f"job_{payload['job_id']}"
    payload["run_id"] = run_id
    work_dir = Path(os.getenv("THETA_WORK_DIR", "/tmp/theta-worker")) / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "worker.log"

    _callback(payload, "running")
    _download_input(payload)
    env = _build_common_env(payload, work_dir)
    for cmd in _pipeline_commands(payload):
        _run_command(cmd, env, log_path)
    uploaded = _upload_outputs(payload, work_dir, log_path)
    _callback(payload, "succeeded")
    return {"success": True, "run_id": run_id, "uploaded_files": uploaded}


def _run_redis_loop() -> None:
    import redis

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise RuntimeError("REDIS_URL is required for --redis-loop")
    queue = os.getenv("REDIS_TRAINING_QUEUE", "theta:training:jobs")
    client = redis.from_url(redis_url, decode_responses=True)
    print(f"[redis] consuming queue {queue}", flush=True)
    while True:
        _, raw = client.brpop(queue)
        payload = json.loads(raw)
        try:
            result = run_job(payload)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception:
            traceback.print_exc()


def main() -> None:
    args = _parse_args()
    if args.redis_loop:
        _run_redis_loop()
        return
    payload = _payload_from_args(args)
    try:
        result = run_job(payload)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        traceback.print_exc()
        _callback(payload, "failed", str(exc))
        raise


if __name__ == "__main__":
    main()
