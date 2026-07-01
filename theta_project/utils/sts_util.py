"""
STS 临时凭证服务 - 用于前端直传 OSS

功能:
1. 调用阿里云 STS 服务获取临时访问凭证（TTL 缓存）
2. 生成前端直传所需的上传策略
3. 返回标准化的上传路径
"""

import os
import json
import hashlib
import hmac
import base64
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# 阿里云配置
ALIBABA_CLOUD_ACCESS_KEY_ID = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID")
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
ALIBABA_CLOUD_REGION_ID = os.getenv("ALIBABA_CLOUD_REGION_ID", "cn-shanghai")
OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT")

# STS 配置
STS_ROLE_ARN = os.getenv("STS_ROLE_ARN", "")  # RAM 角色 ARN
STS_DURATION_SECONDS = int(os.getenv("STS_DURATION_SECONDS", "3600"))  # 凭证有效期


def get_sts_client():
    """创建 STS 客户端"""
    try:
        from alibabacloud_sts20150401.client import Client as StsClient
        from alibabacloud_tea_openapi import models as open_api_models
        
        config = open_api_models.Config(
            access_key_id=ALIBABA_CLOUD_ACCESS_KEY_ID,
            access_key_secret=ALIBABA_CLOUD_ACCESS_KEY_SECRET,
            region_id=ALIBABA_CLOUD_REGION_ID
        )
        config.endpoint = f"sts.{ALIBABA_CLOUD_REGION_ID}.aliyuncs.com"
        return StsClient(config)
    except ImportError:
        print("[WARN] alibabacloud_sts20150401 not installed, using fallback")
        return None


# ---------------------------------------------------------------------------
# STS Token TTL 缓存（避免每次上传请求都调用 AssumeRole）
# ---------------------------------------------------------------------------

class _STSTokenCache:
    """线程安全的 TTL 缓存，key = (username, dataset_name)。"""

    def __init__(self, ttl_seconds: int):
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() < expires_at:
                return value
            del self._cache[key]
            return None

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            self._cache[key] = (time.time() + self._ttl, value)


# 缓存 TTL 略小于 STS 令牌有效期，确保取出的令牌始终有效
_sts_cache = _STSTokenCache(ttl_seconds=max(STS_DURATION_SECONDS - 100, 100))


def _build_upload_path(username: str, dataset_name: str) -> str:
    return f"raw_data/{username}/{dataset_name}/"


def get_sts_token(username: str, dataset_name: str) -> Dict[str, Any]:
    """
    获取 STS 临时凭证（TTL 缓存版本）。

    同一 (username, dataset_name) 在 TTL 内直接返回缓存值，
    避免每次上传请求都调用 AssumeRole（网络延迟 ~100-300ms）。
    """
    cache_key = f"{username}:{dataset_name}"
    cached = _sts_cache.get(cache_key)
    if cached is not None:
        return cached

    upload_path = _build_upload_path(username, dataset_name)

    # 尝试使用 STS SDK
    client = get_sts_client()

    
    if client and STS_ROLE_ARN:
        try:
            from alibabacloud_sts20150401 import models as sts_models
            
            # 构建 Policy，限制只能上传到指定路径
            policy = {
                "Version": "1",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "oss:PutObject",
                            "oss:InitiateMultipartUpload",
                            "oss:UploadPart",
                            "oss:CompleteMultipartUpload",
                            "oss:AbortMultipartUpload"
                        ],
                        "Resource": [
                            f"acs:oss:*:*:{OSS_BUCKET_NAME}/{upload_path}*"
                        ]
                    }
                ]
            }
            
            request = sts_models.AssumeRoleRequest(
                role_arn=STS_ROLE_ARN,
                role_session_name=f"theta_upload_{username}_{dataset_name}",
                duration_seconds=STS_DURATION_SECONDS,
                policy=json.dumps(policy)
            )
            
            response = client.assume_role(request)

            if response and response.body and response.body.credentials:
                creds = response.body.credentials
                result = {
                    "credentials": {
                        "access_key_id": creds.access_key_id,
                        "access_key_secret": creds.access_key_secret,
                        "security_token": creds.security_token,
                        "expiration": creds.expiration
                    },
                    "upload_path": upload_path,
                    "bucket": OSS_BUCKET_NAME,
                    "endpoint": OSS_ENDPOINT,
                    "region": ALIBABA_CLOUD_REGION_ID,
                }
                _sts_cache.set(cache_key, result)
                return result
        except Exception as e:
            print(f"[ERROR] STS AssumeRole failed: {e}")

    # 回退方案：使用主账号凭证（仅用于开发测试）
    print("[WARN] Using main account credentials (not recommended for production)")
    expiration = (datetime.utcnow() + timedelta(seconds=STS_DURATION_SECONDS)).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = {
        "credentials": {
            "access_key_id": ALIBABA_CLOUD_ACCESS_KEY_ID,
            "access_key_secret": ALIBABA_CLOUD_ACCESS_KEY_SECRET,
            "security_token": "",
            "expiration": expiration
        },
        "upload_path": upload_path,
        "bucket": OSS_BUCKET_NAME,
        "endpoint": OSS_ENDPOINT,
        "region": ALIBABA_CLOUD_REGION_ID,
    }
    _sts_cache.set(cache_key, result)
    return result


def generate_upload_policy(username: str, dataset_name: str, max_size_mb: int = 500) -> Dict[str, Any]:
    """
    生成 OSS PostObject 上传策略（用于表单直传）
    
    Args:
        username: 用户名
        dataset_name: 数据集名称
        max_size_mb: 最大文件大小 (MB)
        
    Returns:
        {
            "policy": "base64 encoded policy",
            "signature": "signature",
            "access_key_id": "...",
            "upload_path": "raw_data/{username}/{dataset_name}/",
            "host": "https://bucket.oss-cn-shanghai.aliyuncs.com"
        }
    """
    upload_path = f"raw_data/{username}/{dataset_name}/"
    expiration = (datetime.utcnow() + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # 构建 Policy
    policy_dict = {
        "expiration": expiration,
        "conditions": [
            {"bucket": OSS_BUCKET_NAME},
            ["starts-with", "$key", upload_path],
            ["content-length-range", 0, max_size_mb * 1024 * 1024]
        ]
    }
    
    policy_json = json.dumps(policy_dict)
    policy_base64 = base64.b64encode(policy_json.encode()).decode()
    
    # 计算签名
    signature = base64.b64encode(
        hmac.new(
            ALIBABA_CLOUD_ACCESS_KEY_SECRET.encode(),
            policy_base64.encode(),
            hashlib.sha1
        ).digest()
    ).decode()
    
    return {
        "policy": policy_base64,
        "signature": signature,
        "access_key_id": ALIBABA_CLOUD_ACCESS_KEY_ID,
        "upload_path": upload_path,
        "host": f"https://{OSS_BUCKET_NAME}.{OSS_ENDPOINT}",
        "expiration": expiration
    }


def get_oss_file_url(username: str, dataset_name: str, filename: str) -> str:
    """
    获取 OSS 文件的完整路径
    
    Args:
        username: 用户名
        dataset_name: 数据集名称
        filename: 文件名
        
    Returns:
        OSS 路径，如 "raw_data/zhangsan/my_research/data.csv"
    """
    return f"raw_data/{username}/{dataset_name}/{filename}"


def get_result_url(username: str, dataset_name: str, model_type: str, run_id: str, file_path: str = "") -> str:
    """
    获取训练结果的 OSS 路径
    
    Args:
        username: 用户名
        dataset_name: 数据集名称
        model_type: 模型类型
        run_id: 运行 ID
        file_path: 文件相对路径
        
    Returns:
        OSS 路径
    """
    base_path = f"results/{username}/{dataset_name}/{model_type}/{run_id}"
    if file_path:
        return f"{base_path}/{file_path}"
    return base_path
