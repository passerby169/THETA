import json
import os
import uuid
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SubmittedJob:
    external_job_id: str
    run_id: str
    status: str
    message: str = ""


def _redis_client():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None
    import redis

    return redis.from_url(redis_url, decode_responses=True)


def submit_training_job(payload: dict[str, Any]) -> SubmittedJob:
    """Dispatch a THETA training job to the configured GPU provider.

    First production target is a Modal/HTTP endpoint. If no endpoint is
    configured, the job is pushed into Redis for a worker to consume.
    """
    provider = os.getenv("GPU_PROVIDER", "queue").lower()
    run_id = payload.get("run_id") or f"{payload['job_id']}_{uuid.uuid4().hex[:8]}"
    payload = {**payload, "run_id": run_id}

    if provider in {"modal", "modal_http", "http"}:
        endpoint = os.getenv("GPU_TRAINING_ENDPOINT")
        if not endpoint:
            raise RuntimeError("GPU_TRAINING_ENDPOINT is required for GPU_PROVIDER=modal/http")
        token = os.getenv("GPU_TRAINING_TOKEN")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(
                f"GPU provider rejected job: HTTP {response.status_code} {response.text[:500]}"
            )
        body = response.json() if response.content else {}
        return SubmittedJob(
            external_job_id=str(body.get("job_id") or body.get("id") or run_id),
            run_id=str(body.get("run_id") or run_id),
            status=str(body.get("status") or "running"),
            message=str(body.get("message") or "GPU job submitted"),
        )

    redis_client = _redis_client()
    queue_name = os.getenv("REDIS_TRAINING_QUEUE", "theta:training:jobs")
    if redis_client is not None:
        redis_client.lpush(queue_name, json.dumps(payload, ensure_ascii=False))
        return SubmittedJob(
            external_job_id=f"redis:{run_id}",
            run_id=run_id,
            status="pending",
            message=f"Queued in Redis list {queue_name}",
        )

    return SubmittedJob(
        external_job_id=f"local:{run_id}",
        run_id=run_id,
        status="pending",
        message="No GPU endpoint or Redis queue configured; job is recorded but not dispatched",
    )
