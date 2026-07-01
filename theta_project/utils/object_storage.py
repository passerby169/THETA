import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


@dataclass
class StorageEntry:
    key: str
    size: int = 0
    last_modified: Optional[datetime] = None
    is_prefix: bool = False


def storage_provider() -> str:
    return os.getenv("STORAGE_PROVIDER", "r2").lower()


def bucket_name() -> str:
    value = os.getenv("R2_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME") or os.getenv("OSS_BUCKET_NAME")
    if not value:
        raise RuntimeError("Object storage bucket is not configured")
    return value


def public_base_url() -> Optional[str]:
    return os.getenv("R2_PUBLIC_BASE_URL") or os.getenv("S3_PUBLIC_BASE_URL")


def _endpoint_url() -> Optional[str]:
    if storage_provider() == "r2":
        account_id = os.getenv("R2_ACCOUNT_ID")
        explicit = os.getenv("R2_ENDPOINT_URL")
        if explicit:
            return explicit
        if account_id:
            return f"https://{account_id}.r2.cloudflarestorage.com"
    return os.getenv("S3_ENDPOINT_URL")


def _access_key_id() -> Optional[str]:
    return os.getenv("R2_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")


def _secret_access_key() -> Optional[str]:
    return os.getenv("R2_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")


def _region() -> str:
    if storage_provider() == "r2":
        return os.getenv("R2_REGION", "auto")
    return os.getenv("AWS_REGION", "us-east-1")


def client():
    return boto3.client(
        "s3",
        endpoint_url=_endpoint_url(),
        aws_access_key_id=_access_key_id(),
        aws_secret_access_key=_secret_access_key(),
        region_name=_region(),
        config=Config(signature_version="s3v4"),
    )


def raw_data_key(username: str, dataset: str, filename: str) -> str:
    safe_filename = filename.strip().replace("\\", "/").split("/")[-1]
    return f"raw_data/{username}/{dataset}/{safe_filename}"


def results_prefix(username: str, dataset: str) -> str:
    return f"results/{username}/{dataset}/"


def object_url(key: str) -> str:
    base = public_base_url()
    if base:
        return f"{base.rstrip('/')}/{key}"
    endpoint = _endpoint_url()
    if endpoint:
        return f"{endpoint.rstrip('/')}/{bucket_name()}/{key}"
    return f"s3://{bucket_name()}/{key}"


def create_presigned_upload(
    key: str,
    content_type: str = "application/octet-stream",
    expires_in: int = 3600,
) -> dict[str, Any]:
    upload_url = client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": bucket_name(),
            "Key": key,
            "ContentType": content_type,
        },
        ExpiresIn=expires_in,
    )
    return {
        "provider": storage_provider(),
        "bucket": bucket_name(),
        "region": _region(),
        "endpoint": _endpoint_url() or "",
        "object_key": key,
        "upload_path": key.rsplit("/", 1)[0] + "/",
        "upload_url": upload_url,
        "method": "PUT",
        "headers": {"Content-Type": content_type},
        "public_url": object_url(key),
    }


def list_objects(prefix: str, delimiter: Optional[str] = None) -> list[StorageEntry]:
    kwargs: dict[str, Any] = {"Bucket": bucket_name(), "Prefix": prefix}
    if delimiter:
        kwargs["Delimiter"] = delimiter

    entries: list[StorageEntry] = []
    paginator = client().get_paginator("list_objects_v2")
    for page in paginator.paginate(**kwargs):
        for item in page.get("Contents", []):
            entries.append(
                StorageEntry(
                    key=item["Key"],
                    size=item.get("Size", 0),
                    last_modified=item.get("LastModified"),
                    is_prefix=False,
                )
            )
        for item in page.get("CommonPrefixes", []):
            entries.append(StorageEntry(key=item["Prefix"], is_prefix=True))
    return entries


def object_exists(key: str) -> bool:
    try:
        client().head_object(Bucket=bucket_name(), Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def get_object_bytes(key: str) -> bytes:
    response = client().get_object(Bucket=bucket_name(), Key=key)
    return response["Body"].read()


def get_object_text(key: str, encoding: str = "utf-8") -> str:
    return get_object_bytes(key).decode(encoding)


def get_object_json(key: str) -> Any:
    return json.loads(get_object_text(key))


def put_object_bytes(key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
    client().put_object(Bucket=bucket_name(), Key=key, Body=body, ContentType=content_type)


def delete_keys(keys: Iterable[str]) -> int:
    keys = [k for k in keys if k]
    if not keys:
        return 0
    deleted = 0
    for i in range(0, len(keys), 1000):
        chunk = keys[i:i + 1000]
        client().delete_objects(
            Bucket=bucket_name(),
            Delete={"Objects": [{"Key": key} for key in chunk], "Quiet": True},
        )
        deleted += len(chunk)
    return deleted


def delete_prefix(prefix: str) -> int:
    return delete_keys(entry.key for entry in list_objects(prefix) if not entry.is_prefix)
