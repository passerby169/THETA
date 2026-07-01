#!/usr/bin/env python3
"""
用户注册 + 数据库联动 + OSS 联调测试脚本

默认测试项:
1. 用户注册接口
2. 用户登录接口
3. 数据库用户落库验证
4. OSS STS 凭证接口验证
5. 上传完成回写数据库验证

可选测试项:
- --live-oss: 执行真实 OSS 写入/读取/删除探测
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from dotenv import load_dotenv

# 让脚本可从项目根目录外执行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app.database import SessionLocal, User, File  # noqa: E402
from main import app, verify_password  # noqa: E402
from utils.oss_util import get_oss_bucket  # noqa: E402


def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"缺少环境变量: {key}")
    return value


def assert_ok(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_unique_user() -> tuple[str, str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    username = f"regtest_{ts}"
    email = f"{username}@example.com"
    password = "TestPass123!"
    return username, email, password


def check_user_in_db(username: str, plain_password: str) -> User:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        assert_ok(user is not None, "数据库未找到新注册用户")
        assert_ok(user.email.endswith("@example.com"), "用户邮箱格式异常")
        assert_ok(user.hashed_password != plain_password, "密码未加密存储")
        assert_ok(verify_password(plain_password, user.hashed_password), "密码哈希校验失败")
        return user
    finally:
        db.close()


def check_file_record(user_id: int, filename: str, dataset_name: str) -> None:
    db = SessionLocal()
    try:
        expected_path = f"raw_data/regtest/{dataset_name}/{filename}"
        file_row = db.query(File).filter(
            File.owner_id == user_id,
            File.file_path == expected_path,
        ).first()
        assert_ok(file_row is not None, "上传完成后数据库未记录文件")
    finally:
        db.close()


def run_live_oss_probe(prefix: str) -> None:
    bucket = get_oss_bucket()
    key = f"{prefix}/probe.txt"
    payload = b"theta-oss-probe"

    put_result = bucket.put_object(key, payload)
    assert_ok(200 <= put_result.status < 300, "OSS 写入失败")

    get_result = bucket.get_object(key)
    content = get_result.read()
    assert_ok(content == payload, "OSS 读取内容不一致")

    del_result = bucket.delete_object(key)
    assert_ok(200 <= del_result.status < 300, "OSS 删除失败")


def main() -> None:
    parser = argparse.ArgumentParser(description="注册 + 数据库 + OSS 联调测试")
    parser.add_argument("--live-oss", action="store_true", help="执行真实 OSS 读写探测")
    args = parser.parse_args()

    # 基础环境检查
    require_env("DATABASE_URL")
    require_env("OSS_BUCKET_NAME")
    require_env("OSS_ENDPOINT")

    client = TestClient(app)
    username, email, password = make_unique_user()
    dataset_name = "reg_db_oss_test"
    filename = "sample.csv"

    print("=" * 60)
    print("[1/6] 用户注册")
    print("=" * 60)
    register_resp = client.post(
        "/api/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert_ok(register_resp.status_code == 201, f"注册失败: {register_resp.status_code}, {register_resp.text}")
    user_id = register_resp.json()["id"]
    print(f"[OK] 注册成功: username={username}, user_id={user_id}")

    print("\n" + "=" * 60)
    print("[2/6] 数据库落库验证")
    print("=" * 60)
    db_user = check_user_in_db(username, password)
    print(f"[OK] 用户已落库: id={db_user.id}, email={db_user.email}")

    print("\n" + "=" * 60)
    print("[3/6] 用户登录")
    print("=" * 60)
    login_resp = client.post(
        "/api/auth/login",
        data={"username": username, "password": password},
    )
    assert_ok(login_resp.status_code == 200, f"登录失败: {login_resp.status_code}, {login_resp.text}")
    token = login_resp.json().get("access_token")
    assert_ok(bool(token), "登录成功但未返回 access_token")
    auth_headers = {"Authorization": f"Bearer {token}"}
    print("[OK] 登录成功并获取 token")

    print("\n" + "=" * 60)
    print("[4/6] OSS STS 接口验证")
    print("=" * 60)
    sts_resp = client.get(
        "/api/oss/sts-token",
        params={"dataset_name": dataset_name},
        headers=auth_headers,
    )
    assert_ok(sts_resp.status_code == 200, f"STS 接口失败: {sts_resp.status_code}, {sts_resp.text}")
    sts_json = sts_resp.json()
    assert_ok("credentials" in sts_json, "STS 响应缺少 credentials")
    assert_ok("upload_path" in sts_json, "STS 响应缺少 upload_path")
    assert_ok(sts_json["upload_path"].startswith(f"raw_data/{username}/{dataset_name}/"), "upload_path 不符合预期")
    print(f"[OK] STS 返回 upload_path: {sts_json['upload_path']}")

    print("\n" + "=" * 60)
    print("[5/6] 上传完成回写 DB")
    print("=" * 60)
    complete_resp = client.post(
        "/api/upload/complete",
        headers=auth_headers,
        json={
            "dataset_name": dataset_name,
            "filename": filename,
            "oss_path": f"raw_data/{username}/{dataset_name}/{filename}",
            "file_size": 123,
        },
    )
    assert_ok(
        complete_resp.status_code == 201,
        f"upload/complete 失败: {complete_resp.status_code}, {complete_resp.text}",
    )

    # upload/complete 实现内部固定通过 get_oss_file_url 生成路径，不读请求里的 oss_path
    db = SessionLocal()
    try:
        file_row = db.query(File).filter(
            File.owner_id == user_id,
            File.filename == filename,
        ).order_by(File.id.desc()).first()
        assert_ok(file_row is not None, "数据库未写入文件记录")
        assert_ok(
            file_row.file_path == f"raw_data/{username}/{dataset_name}/{filename}",
            f"文件路径异常: {file_row.file_path}",
        )
    finally:
        db.close()
    print("[OK] upload/complete 成功回写数据库")

    print("\n" + "=" * 60)
    print("[6/6] 可选真实 OSS 探测")
    print("=" * 60)
    if args.live_oss:
        probe_prefix = f"raw_data/{username}/{dataset_name}"
        run_live_oss_probe(probe_prefix)
        print("[OK] OSS 真实读写探测通过")
    else:
        print("[SKIP] 未启用 --live-oss，仅验证后端 OSS 接口返回")

    print("\n[SUCCESS] 联调测试全部通过")


if __name__ == "__main__":
    main()
