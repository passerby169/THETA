import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Union, Dict, Any, Optional

from fastapi import FastAPI, Depends, HTTPException, Header, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
import shutil
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from dotenv import load_dotenv

from app.database import Base, engine, get_db, SessionLocal, User, File, TrainingJob, ChatMessage
from services.dlc_service import submit_job, get_job_status
from utils.oss_util import sync_theta_project_to_oss, get_oss_bucket, OSS_ENDPOINT, OSS_BUCKET_NAME
from utils.sts_util import get_sts_token, generate_upload_policy, get_oss_file_url
from utils.prompts import AI_CHAT_SYSTEM_PROMPT, AI_CHAT_SYSTEM_PROMPT_MULTI, DASHSCOPE_MODEL, DASHSCOPE_VL_MODEL, STREAM_RESPONSE

# =============================================================================
# OSS 路径配置 (与 DLC 容器挂载点对应)
# =============================================================================
# DLC 容器挂载:
#   /mnt/raw_data/  <- oss://bucket/raw_data/   (用户上传数据)
#   /mnt/results/   <- oss://bucket/results/    (训练输出)
#   /mnt/models/    <- oss://bucket/models/     (预训练模型)
#
# 后端服务器需要通过 OSS SDK 访问这些路径
OSS_BUCKET = os.getenv("OSS_BUCKET_NAME", "theta-prod-20260123")
OSS_RESULTS_PREFIX = "results"  # 训练结果前缀
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Theta Project API",
    description="Theta 训练平台 API - 提供用户认证、文件上传、模型训练和结果查询等功能",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# =============================================================================
# OSS 查询缓存（线程安全 TTL 缓存，避免重复遍历 OSS）
# =============================================================================

class TTLCache:
    """
    简单的线程安全 TTL 缓存。

    同一 key 在 TTL 有效期内返回缓存值，过期后重新计算。
    避免在 FastAPI 多worker 场景下重复遍历 OSS。
    """

    def __init__(self, ttl_seconds: int = 60):
        self._ttl = ttl_seconds
        self._cache: dict[str, tuple[float, Any]] = {}  # key -> (expires_at, value)
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.time() < expires_at:
                return value
            del self._cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = (time.time() + self._ttl, value)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# 缓存实例：OSS 结果路径（训练产物路径，60s TTL，训练完成后写入新产物才会变化）
# TTL 60s：在同一训练结果的页面刷新中不会重复查 OSS，新训练完成后缓存自动失效
_oss_path_cache = TTLCache(ttl_seconds=60)

# 缓存实例：OSS 数据集列表（5分钟 TTL，数据集增删不频繁）
_oss_dataset_cache = TTLCache(ttl_seconds=300)


# =============================================================================
# OSS 工具函数（带缓存优化）
# =============================================================================

def _oss_result_path_key(username: str, dataset: str, model: str) -> str:
    return f"{username}:{dataset}:{model}"


def _oss_datasets_key(username: str) -> str:
    return f"datasets:{username}"


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class FileResponse(BaseModel):
    id: int
    filename: str
    file_path: str
    file_type: str
    created_at: datetime
    dataset_name: Optional[str] = None  # 从路径提取的数据集名称

    class Config:
        from_attributes = True


class TrainStartRequest(BaseModel):
    file_id: int
    dataset_name: Optional[str] = None
    model_type: str = "lda"
    model_size: str = "0.6B"
    mode: str = "zero_shot"
    num_topics: int = 20
    epochs: int = 100
    batch_size: Optional[int] = 64
    learning_rate: Optional[float] = 0.002
    hidden_dim: Optional[int] = 512
    patience: Optional[int] = 10
    vocab_size: Optional[int] = 5000
    language: Optional[str] = "chinese"


class STSTokenRequest(BaseModel):
    dataset_name: str


class STSTokenResponse(BaseModel):
    credentials: dict
    upload_path: str
    bucket: str
    endpoint: str
    region: str


class UploadCompleteRequest(BaseModel):
    dataset_name: str
    filename: str
    oss_path: str
    file_size: Optional[int] = None


class TrainingJobResponse(BaseModel):
    id: int
    user_id: int
    status: str
    dlc_job_id: Optional[str]
    run_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    # 训练参数（与 TrainingJob DB 模型保持一致）
    model_type: Optional[str] = None
    model_size: Optional[str] = None
    num_topics: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    hidden_dim: Optional[int] = None
    patience: Optional[int] = None
    vocab_size: Optional[int] = None
    mode: Optional[str] = None
    language: Optional[str] = None

    class Config:
        from_attributes = True


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_user_directories(user_id: int) -> None:
    """Create user directories after successful registration."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    uploads_path = os.path.join(base_path, "users", f"user_{user_id}", "uploads")
    outputs_path = os.path.join(base_path, "users", f"user_{user_id}", "outputs")
    
    try:
        os.makedirs(uploads_path, exist_ok=True)
        os.makedirs(outputs_path, exist_ok=True)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user directories: {str(e)}"
        )


def init_test_user(db: Session):
    """Create a test user if not exists."""
    test_user = db.query(User).filter(
        (User.username == "test") | (User.email == "test@test.com")
    ).first()
    if not test_user:
        hashed_password = get_password_hash("test123")
        test_user = User(
            username="test",
            email="test@test.com",
            hashed_password=hashed_password,
            is_active=True
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        create_user_directories(test_user.id)
        print(f"Test user created: test / test123")
    else:
        print(f"Test user already exists, skipping creation")


# Initialize test user on startup
with SessionLocal() as db:
    init_test_user(db)


@app.post(
    "/api/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="用户注册",
    description="注册新用户账号。注册成功后会自动创建用户专属的 uploads 和 outputs 目录。",
    tags=["认证"]
)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        if existing_user.username == user_data.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    hashed_password = get_password_hash(user_data.password)
    
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        is_active=True
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    create_user_directories(new_user.id)
    
    return new_user


@app.post(
    "/api/auth/login",
    response_model=Token,
    summary="用户登录",
    description="使用用户名和密码登录，返回 JWT Token。Token 包含用户邮箱信息，有效期30分钟。",
    tags=["认证"]
)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and return JWT token."""
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "email": user.email},
        expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@app.get(
    "/api/auth/me",
    response_model=UserResponse,
    summary="获取当前用户信息",
    tags=["认证"]
)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """返回当前登录用户的基本信息。"""
    return current_user


@app.post(
    "/api/auth/logout",
    summary="登出",
    tags=["认证"]
)
def logout():
    """JWT 无状态，登出由前端清除 token。"""
    return {"message": "Logged out successfully"}


class UpdateProfileRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None


@app.put(
    "/api/auth/profile",
    response_model=UserResponse,
    summary="更新用户资料",
    tags=["认证"]
)
def update_profile(
    request: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """更新当前用户的用户名或邮箱。"""
    if request.username:
        existing = db.query(User).filter(User.username == request.username, User.id != current_user.id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already taken")
        current_user.username = request.username
    if request.email:
        existing = db.query(User).filter(User.email == request.email, User.id != current_user.id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        current_user.email = request.email
    db.commit()
    db.refresh(current_user)
    return current_user


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


@app.post(
    "/api/auth/change-password",
    summary="修改密码",
    tags=["认证"]
)
def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """修改当前用户的密码。"""
    if not verify_password(request.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    current_user.hashed_password = get_password_hash(request.new_password)
    db.commit()
    return {"message": "Password changed successfully"}


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def get_current_user_optional(db: Session = Depends(get_db), authorization: Optional[str] = Header(None)) -> Optional[User]:
    """Optional auth - returns None if token invalid instead of throwing. For dev only."""
    if not authorization:
        return None
    try:
        token = authorization.replace("Bearer ", "")
        return get_current_user(token, db)
    except HTTPException:
        return None


def get_test_user(db: Session = Depends(get_db)) -> User:
    """Get test user for bypass mode."""
    user = db.query(User).filter(User.username == "test").first()
    if not user:
        raise HTTPException(status_code=401, detail="Test user not found")
    return user


@app.post(
    "/api/upload",
    response_model=FileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文件",
    description="上传训练数据文件。文件将保存到用户专属的 uploads 目录，并在数据库中记录文件信息。",
    tags=["文件管理"]
)
def upload_file(
    file: UploadFile,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a file for the current user."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(base_path, "users", f"user_{current_user.id}", "uploads")
    
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
    
    db_file = File(
        owner_id=current_user.id,
        filename=file.filename,
        file_path=file_path,
        file_type="uploads",
        created_at=datetime.utcnow()
    )
    
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    return db_file


@app.post(
    "/api/upload/test",
    response_model=FileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="测试上传文件 (无需认证)",
    description="测试用上传接口，使用内置测试用户。",
    tags=["测试"]
)
def upload_file_test(
    file: UploadFile,
    db: Session = Depends(get_db)
):
    """Upload a file for testing (bypasses auth)."""
    test_user = get_test_user(db)
    base_path = os.path.dirname(os.path.abspath(__file__))
    upload_dir = os.path.join(base_path, "users", f"user_{test_user.id}", "uploads")

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

    db_file = File(
        owner_id=test_user.id,
        filename=file.filename,
        file_path=file_path,
        file_type="uploads",
        created_at=datetime.utcnow()
    )

    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    return db_file


@app.get(
    "/api/files",
    response_model=List[FileResponse],
    summary="获取文件列表",
    description="获取当前用户上传的所有文件列表，包含文件ID、文件名、路径和上传时间。",
    tags=["文件管理"]
)
def list_files(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all files for the current user."""
    files = db.query(File).filter(File.owner_id == current_user.id).all()

    result = []
    for f in files:
        # 跳过没有有效 dataset_name 的文件（这些是脏数据）
        if not f.file_path or not f.file_path.startswith("raw_data/"):
            continue

        parts = f.file_path.split("/")
        if len(parts) < 3:
            continue

        dataset_name = parts[2]
        result.append({
            "id": f.id,
            "filename": f.filename,
            "file_path": f.file_path,
            "file_type": f.file_type,
            "created_at": f.created_at,
            "dataset_name": dataset_name
        })
    return result


@app.get(
    "/api/oss/sts-token",
    response_model=STSTokenResponse,
    summary="获取 STS 临时凭证",
    description="获取 OSS 前端直传所需的 STS 临时凭证。前端使用此凭证直接上传文件到 OSS，无需经过后端中转。",
    tags=["文件管理"]
)
def get_sts_token_api(
    dataset_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    获取 STS 临时凭证用于前端直传 OSS
    
    返回:
    - credentials: 临时访问凭证 (AccessKeyId, AccessKeySecret, SecurityToken)
    - upload_path: 上传路径 raw_data/{username}/{dataset_name}/
    - bucket: OSS Bucket 名称
    - endpoint: OSS Endpoint
    """
    try:
        sts_result = get_sts_token(current_user.username, dataset_name)
        return STSTokenResponse(**sts_result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get STS token: {str(e)}"
        )


@app.post(
    "/api/upload/complete",
    response_model=FileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="通知上传完成",
    description="前端直传 OSS 完成后，调用此接口通知后端记录文件信息。",
    tags=["文件管理"]
)
def upload_complete(
    request: UploadCompleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    前端直传完成后通知后端
    
    记录文件信息到数据库，OSS 路径格式: raw_data/{username}/{dataset_name}/{filename}
    """
    # 构建 OSS 路径
    oss_path = get_oss_file_url(current_user.username, request.dataset_name, request.filename)
    
    # 检查是否已存在
    existing_file = db.query(File).filter(
        File.owner_id == current_user.id,
        File.file_path == oss_path
    ).first()
    
    if existing_file:
        # 更新已有记录
        existing_file.created_at = datetime.utcnow()
        db.commit()
        db.refresh(existing_file)
        return existing_file
    
    # 创建新记录
    db_file = File(
        owner_id=current_user.id,
        filename=request.filename,
        file_path=oss_path,
        file_type="oss_upload",
        created_at=datetime.utcnow()
    )
    
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    return db_file


# ========== 预处理 ==========

import uuid
from datetime import datetime

# 内存中存储预处理任务状态（实际项目中应该用数据库）
preprocessing_jobs: Dict[str, dict] = {}


class PreprocessingStatusResponse(BaseModel):
    dataset: Optional[str] = None
    has_bow: bool
    has_embeddings: bool
    ready_for_training: bool
    bow_path: Optional[str] = None
    embedding_path: Optional[str] = None
    vocab_path: Optional[str] = None


class PreprocessingJobResponse(BaseModel):
    job_id: str
    dataset: str
    status: str
    progress: int
    message: Optional[str] = None
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    bow_path: Optional[str] = None
    embedding_path: Optional[str] = None
    vocab_path: Optional[str] = None


class StartPreprocessingRequest(BaseModel):
    dataset: str
    text_column: Optional[str] = None


@app.get(
    "/api/preprocessing/check/{dataset}",
    response_model=PreprocessingStatusResponse,
    summary="检查预处理状态",
    description="检查数据集是否已完成预处理（BOW 矩阵和嵌入向量）",
    tags=["预处理"]
)
def check_preprocessing_status(
    dataset: str,
    current_user: User = Depends(get_current_user),
):
    """
    检查数据集的预处理状态
    通过检查 OSS 上的文件来判断是否已完成预处理
    """
    import oss2
    bucket = get_oss_bucket()

    # 检查 workspace 目录下的文件
    base_prefix = f"raw_data/{current_user.username}/{dataset}/workspace/"

    has_bow = False
    has_embeddings = False
    has_vocab = False
    bow_path = None
    embedding_path = None
    vocab_path = None

    for obj in oss2.ObjectIterator(bucket, prefix=base_prefix):
        key = obj.key.lower()
        if key.endswith('/'):
            continue
        if 'bow_matrix.npy' in key or 'bow.npy' in key:
            has_bow = True
            bow_path = obj.key
            if has_embeddings and has_vocab:
                break  # 全部找到，提前结束扫描
        elif 'embeddings.npy' in key or 'embedding.npy' in key:
            has_embeddings = True
            embedding_path = obj.key
            if has_bow and has_vocab:
                break
        elif 'vocab.json' in key or 'vocab.txt' in key:
            has_vocab = True
            vocab_path = obj.key
            if has_bow and has_embeddings:
                break

    ready_for_training = has_bow and has_embeddings and has_vocab

    return PreprocessingStatusResponse(
        dataset=dataset,
        has_bow=has_bow,
        has_embeddings=has_embeddings,
        ready_for_training=ready_for_training,
        bow_path=bow_path,
        embedding_path=embedding_path,
        vocab_path=vocab_path
    )


@app.post(
    "/api/preprocessing/start",
    response_model=PreprocessingJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="启动预处理任务",
    description="为数据集启动预处理任务，生成 BOW 矩阵和嵌入向量",
    tags=["预处理"]
)
def start_preprocessing(
    request: StartPreprocessingRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    启动预处理任务
    注意：实际预处理在训练时自动完成，这里只是模拟接口
    """
    job_id = f"prep_{uuid.uuid4().hex[:12]}"
    dataset = request.dataset

    # 先检查是否已经预处理完成
    import oss2
    bucket = get_oss_bucket()
    base_prefix = f"raw_data/{current_user.username}/{dataset}/workspace/"

    has_bow = False
    has_embeddings = False
    has_vocab = False

    for obj in oss2.ObjectIterator(bucket, prefix=base_prefix):
        key = obj.key.lower()
        if 'bow_matrix.npy' in key or 'bow.npy' in key:
            has_bow = True
        elif 'embeddings.npy' in key or 'embedding.npy' in key:
            has_embeddings = True
        elif 'vocab.json' in key or 'vocab.txt' in key:
            has_vocab = True
        if has_bow and has_embeddings and has_vocab:
            break  # 全部找到，提前结束

    if has_bow and has_embeddings and has_vocab:
        # 已经完成预处理
        job = {
            "job_id": job_id,
            "dataset": dataset,
            "status": "completed",
            "progress": 100,
            "message": "预处理已完成",
            "current_stage": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
    else:
        # 模拟预处理过程（实际由训练时完成）
        job = {
            "job_id": job_id,
            "dataset": dataset,
            "status": "completed",
            "progress": 100,
            "message": "预处理已完成（BOW 和嵌入将在训练时自动生成）",
            "current_stage": None,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

    preprocessing_jobs[job_id] = job
    return PreprocessingJobResponse(**job)


@app.get(
    "/api/preprocessing/{job_id}",
    response_model=PreprocessingJobResponse,
    summary="获取预处理任务状态",
    description="查询预处理任务的当前状态",
    tags=["预处理"]
)
def get_preprocessing_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    获取预处理任务状态
    """
    if job_id not in preprocessing_jobs:
        # 如果任务不存在，返回完成状态（兼容已完成的旧任务）
        return PreprocessingJobResponse(
            job_id=job_id,
            dataset="",
            status="completed",
            progress=100,
            message="任务已完成或不存在",
        )

    return PreprocessingJobResponse(**preprocessing_jobs[job_id])


@app.post(
    "/api/train/start",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="启动训练任务",
    description="为指定文件启动 DLC 训练任务。会先同步训练代码到 OSS，然后提交任务到阿里云 PAI-DLC。",
    tags=["训练任务"]
)
def start_training(
    request: TrainStartRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a training job for the specified file."""
    file_record = db.query(File).filter(
        File.id == request.file_id,
        File.owner_id == current_user.id
    ).first()
    
    if not file_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or access denied"
        )
    
    try:
        sync_theta_project_to_oss()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync code to OSS: {str(e)}"
        )
    
    training_job = TrainingJob(
        user_id=current_user.id,
        file_id=request.file_id,
        model_type=request.model_type,
        model_size=request.model_size,
        num_topics=request.num_topics,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        hidden_dim=request.hidden_dim,
        patience=request.patience,
        vocab_size=request.vocab_size,
        mode=request.mode,
        language=request.language,
        status="pending",
        created_at=datetime.utcnow()
    )
    db.add(training_job)
    db.commit()
    db.refresh(training_job)
    
    try:
        dlc_job_id, run_id = submit_job(
            user_id=current_user.id,
            username=current_user.username,
            file_id=request.file_id,
            file_path=file_record.file_path,
            job_id=training_job.id,
            dataset_name=request.dataset_name,
            model_type=request.model_type,
            model_size=request.model_size,
            mode=request.mode,
            num_topics=request.num_topics,
            epochs=request.epochs,
            batch_size=request.batch_size or 64,
            learning_rate=request.learning_rate or 0.002,
            hidden_dim=request.hidden_dim or 512,
            patience=request.patience or 10,
            language=request.language or "chinese",
            vocab_size=request.vocab_size or 5000,
        )

        training_job.dlc_job_id = dlc_job_id
        training_job.run_id = run_id
        if dlc_job_id:
            training_job.status = "running"
        else:
            training_job.status = "failed"
            training_job.error_message = "DLC 任务提交失败，未返回 job_id"
        db.commit()
        db.refresh(training_job)
        
    except Exception as e:
        training_job.status = "failed"
        training_job.error_message = str(e)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit DLC job: {str(e)}"
        )
    
    return training_job


class TrainingMetricsResponse(BaseModel):
    job_id: int
    status: str
    epochs: List[int]
    loss: List[float]
    accuracy: List[float]


class TrainingStatusResponse(BaseModel):
    job_id: int
    status: str
    dlc_job_id: Optional[str]
    error_message: Optional[str] = None
    created_at: datetime
    message: str


class TrainingSummaryResponse(BaseModel):
    job_id: int
    status: str
    summary: Optional[str]


class TrainingCallbackRequest(BaseModel):
    job_id: int
    status: str
    run_id: Optional[str] = None
    secret_key: Optional[str] = None


class TrainingCallbackResponse(BaseModel):
    success: bool
    message: str


def get_training_job_for_user(job_id: int, user_id: int, db: Session) -> TrainingJob:
    """Get training job and verify ownership."""
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == user_id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found or access denied"
        )
    return job


@app.get(
    "/api/train/{job_id}/metrics",
    response_model=TrainingMetricsResponse,
    summary="获取训练指标",
    description="获取训练任务的指标数据，包含每个 epoch 的 loss 和 accuracy，适合前端绑定图表展示。",
    tags=["训练任务"]
)
def get_training_metrics(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get training metrics for a job from OSS."""
    job = get_training_job_for_user(job_id, current_user.id, db)

    # 从数据库获取任务关联的 dataset_name 和 model_type
    file_record = db.query(File).filter(File.id == job.file_id).first() if job.file_id else None
    dataset_name = ""
    if file_record and file_record.file_path:
        # 从 file_path 提取 dataset_name: raw_data/{username}/{dataset_name}/...
        parts = file_record.file_path.split("/")
        if len(parts) >= 3:
            dataset_name = parts[2] if parts[0] == "raw_data" else parts[-2]

    # 从 OSS 读取 training_log.json - DLC 容器实时写入 OSS
    import oss2
    bucket = get_oss_bucket()

    epochs = []
    loss = []
    accuracy = []

    # DLC 输出路径固定结构: results/{username}/{dataset}/{model_type}/{run_id}/training_log.json
    # run_id 由 DLC 容器生成，必须通过 DB 回填才能准确拼接
    possible_paths = []
    if job.run_id:
        possible_paths.append(
            f"results/{current_user.username}/{dataset_name}/{job.model_type}/{job.run_id}/training_log.json"
        )
    # 兜底：兼容旧路径（run_id 尚未回填的情况）
    possible_paths.extend([
        f"results/{current_user.username}/{dataset_name}/{job.model_type}/{job.id}/training_log.json",
        f"results/{current_user.username}/{dataset_name}/training_log.json",
    ])

    def _try_read_log(oss_key: str) -> dict | None:
        """尝试读取单个 OSS 路径，返回解析后的 log_data 或 None。"""
        try:
            result = bucket.get_object(oss_key)
            content = result.read().decode('utf-8')
            return json.loads(content)
        except (oss2.exceptions.NoSuchKey, Exception):
            return None

    log_data = None
    if possible_paths:
        with ThreadPoolExecutor(max_workers=len(possible_paths)) as executor:
            for result_future in executor.map(_try_read_log, possible_paths):
                if result_future is not None:
                    log_data = result_future
                    break  # 第一个成功即停止（节省后续网络开销）

    if log_data:
        if isinstance(log_data, dict) and "metrics" in log_data:
            for entry in log_data["metrics"]:
                epochs.append(entry.get("epoch", 0))
                loss.append(entry.get("loss", 0.0))
                accuracy.append(entry.get("accuracy", 0.0) if entry.get("accuracy") is not None else 0.0)
        elif isinstance(log_data, list):
            for entry in log_data:
                epochs.append(entry.get("epoch", 0))
                loss.append(entry.get("loss", 0.0))
                accuracy.append(entry.get("accuracy", 0.0) if entry.get("accuracy") is not None else 0.0)

    return TrainingMetricsResponse(
        job_id=job_id,
        status=job.status,
        epochs=epochs,
        loss=loss,
        accuracy=accuracy
    )


@app.get(
    "/api/train/{job_id}/status",
    response_model=TrainingStatusResponse,
    summary="查询任务状态",
    description="查询训练任务的实时状态。如果任务处于 running 状态，会主动调用阿里云 SDK 同步最新状态。",
    tags=["训练任务"]
)
def get_training_status(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get real-time status of a training job."""
    job = get_training_job_for_user(job_id, current_user.id, db)
    
    if job.status in ("pending", "creating", "running") and job.dlc_job_id:
        dlc_status = get_job_status(job.dlc_job_id)
        if dlc_status and dlc_status != job.status:
            job.status = dlc_status
            db.commit()
            db.refresh(job)
    
    status_messages = {
        "pending": "任务已创建，等待调度",
        "creating": "任务创建中，等待调度",
        "running": "训练进行中",
        "succeeded": "训练已完成",
        "failed": "训练失败"
    }
    
    return TrainingStatusResponse(
        job_id=job_id,
        status=job.status,
        dlc_job_id=job.dlc_job_id,
        created_at=job.created_at,
        message=status_messages.get(job.status, "未知状态"),
        error_message=job.error_message
    )


@app.get(
    "/api/train/{job_id}/summary",
    response_model=TrainingSummaryResponse,
    summary="获取训练摘要",
    description="获取训练完成后的模型摘要，包含完成时间、总轮次、最终 Loss 和 Accuracy。",
    tags=["训练任务"]
)
def get_training_summary(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get training summary for a completed job."""
    job = get_training_job_for_user(job_id, current_user.id, db)
    
    # 从数据库获取任务关联的 dataset_name
    file_record = db.query(File).filter(File.id == job.file_id).first() if job.file_id else None
    dataset_name = ""
    if file_record and file_record.file_path:
        parts = file_record.file_path.split("/")
        if len(parts) >= 3:
            dataset_name = parts[2] if parts[0] == "raw_data" else parts[-2]
    
    # DLC 输出路径固定结构: results/{username}/{dataset}/{model_type}/{run_id}/
    import oss2
    bucket = get_oss_bucket()

    summary_content = None
    oss_key = None
    if job.run_id:
        oss_key = f"results/{current_user.username}/{dataset_name}/{job.model_type}/{job.run_id}/training_log.json"

    if oss_key:
        try:
            result = bucket.get_object(oss_key)
            content = result.read().decode("utf-8")
            log_data = json.loads(content)
            lines = [
                f"Status: {log_data.get('status', 'unknown')}",
                f"Completed at: {log_data.get('completed_at', 'N/A')}",
            ]
            cfg = log_data.get("config", {})
            if cfg:
                lines.append(f"Topics: {cfg.get('num_topics')}  |  Epochs: {cfg.get('epochs')}  |  Mode: {cfg.get('mode')}")
                lines.append(f"Model size: {cfg.get('model_size')}  |  Vocab size: {cfg.get('vocab_size')}")
            summary_content = "\n".join(lines)
        except oss2.exceptions.NoSuchKey:
            pass
        except Exception as e:
            print(f"[WARN] Failed to read summary from OSS {oss_key}: {e}")
    
    return TrainingSummaryResponse(
        job_id=job_id,
        status=job.status,
        summary=summary_content
    )


@app.post(
    "/api/train/callback",
    response_model=TrainingCallbackResponse,
    summary="训练回调接口",
    description="DLC 训练脚本完成后调用此接口通知服务器更新任务状态。需要提供 job_id 和 status (succeeded/failed)。",
    tags=["训练任务"]
)
def training_callback(
    request: TrainingCallbackRequest,
    db: Session = Depends(get_db)
):
    """Callback endpoint for DLC training script to report completion."""
    if request.secret_key != SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid secret key"
        )
    
    job = db.query(TrainingJob).filter(TrainingJob.id == request.job_id).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )
    
    if request.status not in ["succeeded", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid status. Must be 'succeeded' or 'failed'"
        )
    
    job.status = request.status
    if request.run_id:
        job.run_id = request.run_id
    db.commit()
    
    return TrainingCallbackResponse(
        success=True,
        message=f"Job {request.job_id} status updated to {request.status}"
    )


@app.get(
    "/api/train/jobs",
    response_model=List[TrainingJobResponse],
    summary="Get training jobs list",
    description="Get all training jobs for the current user to restore task status on page reload.",
    tags=["训练任务"]
)
def get_training_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all training jobs for the current user."""
    jobs = db.query(TrainingJob).filter(
        TrainingJob.user_id == current_user.id
    ).order_by(TrainingJob.created_at.desc()).all()
    
    return [
        TrainingJobResponse(
            id=job.id,
            user_id=job.user_id,
            status=job.status,
            dlc_job_id=job.dlc_job_id,
            run_id=job.run_id,
            error_message=job.error_message,
            created_at=job.created_at,
            model_type=job.model_type,
            model_size=job.model_size,
            num_topics=job.num_topics,
            epochs=job.epochs,
            batch_size=job.batch_size,
            learning_rate=job.learning_rate,
            hidden_dim=job.hidden_dim,
            patience=job.patience,
            vocab_size=job.vocab_size,
            mode=job.mode,
            language=job.language,
        )
        for job in jobs
    ]



# ==================== AI Chat ====================

class ImagePayload(BaseModel):
    name: str
    mimeType: str
    size: int
    dataUrl: str

class FilePayload(BaseModel):
    name: str
    mimeType: str
    size: int
    dataUrl: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[dict] = None
    images: Optional[list[ImagePayload]] = None
    files: Optional[list[FilePayload]] = None


class ChatResponse(BaseModel):
    message: str
    thinking: Optional[str] = None


class ChatMessageCreate(BaseModel):
    role: str
    content: str
    session_id: str = "default"


class ChatMessageResponse(BaseModel):
    id: int
    role: str
    content: str
    session_id: str
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessageResponse]


@app.post(
    "/api/agent/chat",
    response_model=ChatResponse,
    summary="AI Chat",
    description="AI chat endpoint using DashScope API (auth optional for dev)",
    tags=["AI Chat"]
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """AI Chat endpoint using Alibaba DashScope with history saving. Supports multimodal (images/files)."""
    # Dev bypass: use test user if not authenticated
    if current_user is None:
        current_user = get_test_user(db)

    # Save user message
    user_msg = ChatMessage(
        created_at=datetime.utcnow(),
        user_id=current_user.id,
        session_id=request.session_id,
        role="user",
        content=request.message
    )
    db.add(user_msg)
    db.flush()  # 写入 DB 但保持事务开启，AI 生成期间不阻塞其他请求

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DashScope API key not configured"
        )

    try:
        import dashscope
        dashscope.api_key = dashscope_api_key

        # Build context
        context = request.context or {}
        user_context = ""
        if context:
            if context.get("current_page"):
                user_context += "\n当前页面: " + str(context.get("current_page"))
            if context.get("projects"):
                user_context += "\n项目列表: " + str(context.get("projects"))
            if context.get("job_status"):
                user_context += "\n当前任务状态: " + str(context.get("job_status"))
            if context.get("current_operation"):
                user_context += "\n当前操作: " + str(context.get("current_operation"))
            if context.get("dataset"):
                user_context += "\n当前数据集: " + str(context.get("dataset"))

        has_images = request.images and len(request.images) > 0
        has_files = request.files and len(request.files) > 0

        if has_images:
            # Use multimodal Vision model
            system_prompt = AI_CHAT_SYSTEM_PROMPT_MULTI
            content_parts = []
            for img in request.images:
                content_parts.append({"image": img.dataUrl})
            user_question = "\n\n用户问题: " + request.message if request.message else "\n\n请分析以上图片内容"
            content_parts.append({"text": system_prompt + user_context + user_question})

            response = dashscope.MultiModalConversation.call(
                model=DASHSCOPE_VL_MODEL,
                messages=[{
                    "role": "user",
                    "content": content_parts
                }]
            )
        else:
            system_prompt = AI_CHAT_SYSTEM_PROMPT
            file_content = ""
            if has_files:
                file_content = "\n\n用户上传了以下文件：\n"
                for f in request.files:
                    file_content += "- " + f.name + " (类型: " + f.mimeType + ")\n"

            full_prompt = system_prompt + user_context + file_content + "\n\n用户问题: " + request.message

            response = dashscope.Generation.call(
                model=DASHSCOPE_MODEL,
                prompt=full_prompt,
                stream=False,
                return_full_text=True,
            )

        if response.status_code == 200:
            if has_images:
                choices = response.output.get("choices", [])
                if choices:
                    msg_content = choices[0].get("message", {}).get("content", [])
                    if isinstance(msg_content, list) and len(msg_content) > 0:
                        reply = msg_content[0].get("text", "")
                    else:
                        reply = str(msg_content) if msg_content else ""
                else:
                    reply = ""
            else:
                output = response.output
                reply = output.get("text", "") if isinstance(output, dict) else str(output)

            if not reply:
                reply = "抱歉，我无法生成回复。"

            ai_msg = ChatMessage(
                created_at=datetime.utcnow(),
                user_id=current_user.id,
                session_id=request.session_id,
                role="ai",
                content=reply
            )
            db.add(ai_msg)
            db.commit()
            return ChatResponse(message=reply, thinking=None)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="DashScope API error: " + str(response.message)
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DashScope API error: " + str(e)
        )


@app.get("/api/chat/history/{session_id}", response_model=ChatHistoryResponse)
def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get chat history for a session."""
    messages = db.query(ChatMessage).filter(
        ChatMessage.user_id == current_user.id,
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.asc()).all()

    return ChatHistoryResponse(messages=[
        ChatMessageResponse(
            id=m.id,
            role=m.role,
            content=m.content,
            session_id=m.session_id,
            created_at=m.created_at
        ) for m in messages
    ])


@app.post("/api/chat/history/{session_id}", response_model=ChatMessageResponse)
def save_chat_message(
    session_id: str,
    message: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save a chat message to history."""
    db_message = ChatMessage(
        created_at=datetime.utcnow(),
        user_id=current_user.id,
        session_id=session_id,
        role=message.role,
        content=message.content
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)

    return ChatMessageResponse(
        id=db_message.id,
        role=db_message.role,
        content=db_message.content,
        session_id=db_message.session_id,
        created_at=db_message.created_at
    )


# ==================== 结果查询 ====================

class TopicWordsResponse(BaseModel):
    dataset: str
    model: str
    topics: Dict[str, list]  # {"0": [["word", weight], ...], ...}


class MetricsResponse(BaseModel):
    dataset: str
    model: str
    metrics: Dict[str, Any]  # TD, iRBO, NPMI, C_V, PPL, etc. (values can be float, list, or str)


class VisualizationFile(BaseModel):
    name: str
    path: str
    url: str
    size: int
    type: str  # "global" or "topic"


class VisualizationResponse(BaseModel):
    dataset: str
    model: str
    global_files: List[VisualizationFile]
    topic_files: Dict[str, List[VisualizationFile]]  # {"1": [files], ...}


class OssDatasetsResponse(BaseModel):
    datasets: List[dict]  # [{name: string, chart_count: number}, ...]


@app.get(
    "/api/data/oss-datasets",
    summary="获取有可视化结果的数据集列表",
    description="扫描 OSS results 目录，返回所有有可视化图表的数据集及其图表数量",
    tags=["数据查询"]
)
def list_oss_datasets(
    current_user: User = Depends(get_current_user),
):
    """
    列出 OSS 上所有有可视化结果的数据集。
    结果缓存 5 分钟（TTL），避免每次请求都全量扫描 OSS。
    """
    import oss2
    cache_key = _oss_datasets_key(current_user.username)
    cached = _oss_dataset_cache.get(cache_key)
    if cached is not None:
        return cached

    bucket = get_oss_bucket()
    username = current_user.username
    base_prefix = f"results/{username}/"

    # 第一步：扫描所有数据集名称（单次 OSS 调用）
    dataset_names: list[str] = []
    for obj in oss2.ObjectIterator(bucket, prefix=base_prefix, delimiter='/'):
        if obj.key.endswith('/'):
            relative = obj.key[len(base_prefix):].rstrip('/')
            if '/' not in relative:
                dataset_names.append(relative)

    if not dataset_names:
        result = OssDatasetsResponse(datasets=[])
        _oss_dataset_cache.set(cache_key, result)
        return result

    # 第二步：并发查询每个数据集的 chart 数量（避免 O(N*M) 串行）
    def _count_charts(ds_name: str) -> tuple[str, int] | None:
        viz_prefix = f"results/{username}/{ds_name}/"
        count = 0
        for viz_obj in oss2.ObjectIterator(bucket, prefix=viz_prefix):
            k = viz_obj.key.lower()
            if "visualization" in k and k.endswith(('.png', '.jpg', '.jpeg')):
                count += 1
        return (ds_name, count) if count > 0 else None

    datasets_map: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=min(len(dataset_names), 8)) as executor:
        for item in executor.map(_count_charts, dataset_names):
            if item is not None:
                datasets_map[item[0]] = item[1]

    result = OssDatasetsResponse(
        datasets=[{"name": n, "chart_count": c} for n, c in sorted(datasets_map.items())]
    )
    _oss_dataset_cache.set(cache_key, result)
    return result


class AvailableModelsResponse(BaseModel):
    dataset: str
    models: List[str]  # ["theta", "nvdm", "bertopic", ...]


@app.get(
    "/api/results/{dataset}/models",
    response_model=AvailableModelsResponse,
    summary="获取可用的模型列表",
    description="扫描 OSS 结果目录，返回该数据集下所有有结果的模型",
    tags=["结果查询"]
)
def get_available_models(
    dataset: str,
    current_user: User = Depends(get_current_user),
):
    """
    扫描 OSS 上的 results 目录，返回该数据集下所有有结果的模型

    支持两种目录结构:
    - 单模型: results/{user}/{dataset}/{model}/{run_id}/
    - 多模型: results/{user}/{dataset}/{run_id}/{model}/
    """
    import oss2

    bucket = get_oss_bucket()
    prefix = f"results/{current_user.username}/{dataset}/"
    models_set = set()

    # 列出 dataset 下的所有目录
    for obj in oss2.ObjectIterator(bucket, prefix=prefix, delimiter='/'):
        if not obj.key.endswith('/'):
            continue

        relative = obj.key[len(prefix):].rstrip('/')
        if not relative:
            continue

        # 检查是 model 名还是 run_id
        # 已知模型列表
        known_models = {"theta", "nvdm", "bertopic", "lda", "hdp", "stm", "btm", "ctm", "etm", "dtm", "prodlda", "gsm"}

        if relative in known_models:
            # 单模型结构: results/{user}/{dataset}/{model}/
            models_set.add(relative)
        else:
            # 可能是 run_id，检查子目录（找到第一个模型即停止扫描）
            for sub_obj in oss2.ObjectIterator(bucket, prefix=obj.key, delimiter='/'):
                if not sub_obj.key.endswith('/'):
                    continue
                sub_relative = sub_obj.key[len(obj.key):].rstrip('/')
                if sub_relative in known_models:
                    models_set.add(sub_relative)
                    break  # 该 run_id 下只需找到一个模型即可

    return AvailableModelsResponse(
        dataset=dataset,
        models=sorted(list(models_set))
    )


def find_result_path(bucket, username: str, dataset: str, model: str) -> str:
    """
    查找训练结果的 OSS 路径，支持两种路径结构:
    - 单模型结构: results/{user}/{dataset}/{model}/{run_id}/
      示例: results/test/dandu_no_theta/ctm/20260410_xxx/
    - 多模型结构: results/{user}/{dataset}/{run_id}/{model}/
      示例: results/test/dandu_theta/20260410_xxx/theta/

    Returns:
        结果路径前缀 (不含尾部/), 如 "results/test/test5/20260410_xxx/theta"
        如果未找到则返回 None
    """
    import oss2

    cache_key = _oss_result_path_key(username, dataset, model)
    cached = _oss_path_cache.get(cache_key)
    if cached is not None:
        return cached

    # 模式1: 单模型结构 results/{user}/{dataset}/{model}/{run_id}/（优先，更精确）
    prefix1 = f"results/{username}/{dataset}/{model}/"
    run_ids1 = []
    for obj in oss2.ObjectIterator(bucket, prefix=prefix1, delimiter='/'):
        if obj.key.endswith('/'):
            run_id = obj.key.replace(prefix1, '').rstrip('/')
            if run_id and not run_id.startswith(model):
                run_ids1.append(run_id)

    # 找到则直接返回，不继续扫描模式2
    if run_ids1:
        result = f"results/{username}/{dataset}/{model}/{sorted(run_ids1)[-1]}"
        _oss_path_cache.set(cache_key, result)
        return result

    # 模式2: 多模型结构 results/{user}/{dataset}/{run_id}/{model}/（扫描开销更大）
    prefix2 = f"results/{username}/{dataset}/"
    run_ids2 = []
    for obj in oss2.ObjectIterator(bucket, prefix=prefix2, delimiter='/'):
        if obj.key.endswith('/'):
            potential = obj.key.replace(prefix2, '').rstrip('/')
            if ',' in potential or potential == model:
                continue
            model_prefix = f"results/{username}/{dataset}/{potential}/{model}/"
            for check_obj in oss2.ObjectIterator(bucket, prefix=model_prefix, delimiter='/'):
                if check_obj.key.endswith('/'):
                    run_ids2.append(potential)
                    break  # 找到一个 run_id 即足够

    if not run_ids2:
        _oss_path_cache.set(cache_key, None)
        return None

    result = f"results/{username}/{dataset}/{sorted(run_ids2)[-1]}/{model}"
    _oss_path_cache.set(cache_key, result)
    return result


@app.get(
    "/api/results/{dataset}/topic-words",
    response_model=TopicWordsResponse,
    summary="获取主题词",
    description="从 OSS 读取指定数据集的 topic_words.json，支持不同模型的不同文件名格式",
    tags=["结果查询"]
)
def get_topic_words(
    dataset: str,
    model: str = "theta",
    current_user: User = Depends(get_current_user),
):
    """
    读取 OSS 上的 topic_words 文件
    支持多种文件名格式:
    - topic_words.json (theta, lda, etc.)
    - topic_words_k{K}.json (bertopic)
    - 主题表.csv (nvdm) - 从可视化结果读取
    支持两种路径结构:
    - 单模型: results/{username}/{dataset}/{model}/{run_id}/model/topic_words.json
    - 多模型: results/{username}/{dataset}/{run_id}/{model}/model/topic_words.json
    """
    import oss2
    import csv
    import io

    bucket = get_oss_bucket()

    # 使用通用路径查找函数
    result_path = find_result_path(bucket, current_user.username, dataset, model)

    if not result_path:
        raise HTTPException(status_code=404, detail=f"未找到数据集 {dataset} 模型 {model} 的训练结果")

    model_prefix = f"{result_path}/model/"

    # 尝试多种可能的 topic_words 文件名
    topic_words_files = []

    # 首先列出 model 目录下的所有文件，找到 topic_words 相关文件
    for obj in oss2.ObjectIterator(bucket, prefix=model_prefix):
        if 'topic_words' in obj.key.lower() and obj.key.endswith('.json'):
            topic_words_files.append(obj.key)

    # 尝试读取 topic_words.json 或变体
    selected_key = None
    if topic_words_files:
        # 按优先级选择：优先 topic_words.json，然后是 topic_words_k*.json
        for key in sorted(topic_words_files):
            if key.endswith('/topic_words.json'):
                selected_key = key
                break
        if not selected_key and topic_words_files:
            selected_key = sorted(topic_words_files)[0]

    topics = None
    if selected_key:
        try:
            result = bucket.get_object(selected_key)
            content = result.read().decode('utf-8')
            topics = json.loads(content)
        except:
            pass

    # 如果没有找到 topic_words 文件，尝试从 主题表.csv 读取（用于 nvdm 等模型）
    if topics is None:
        csv_path = f"{result_path}/visualization/zh/global/主题表.csv"
        try:
            result = bucket.get_object(csv_path)
            content = result.read().decode('utf-8-sig')  # utf-8-sig 处理 BOM
            # 解析 CSV: topic_id, topic_name, strength, keywords
            reader = csv.DictReader(io.StringIO(content))
            topics = {}
            for row in reader:
                topic_id = str(int(row['topic_id']) - 1)  # CSV 是 1-indexed
                keywords_str = row.get('keywords', row.get('keyword', ''))
                if keywords_str:
                    # keywords 格式: "word1, word2, word3, ..."
                    keywords = [kw.strip() for kw in keywords_str.split(',') if kw.strip()]
                    # 转换为 [[word, weight], ...] 格式（权重设为相等）
                    topics[topic_id] = [[kw, 1.0] for kw in keywords[:10]]
        except oss2.exceptions.NoSuchKey:
            pass
        except Exception as e:
            pass

    if topics is None:
        raise HTTPException(status_code=404, detail=f"未找到 {model} 模型的主题词文件")

    return TopicWordsResponse(
        dataset=dataset,
        model=model,
        topics=topics
    )


@app.get(
    "/api/results/{dataset}/metrics",
    response_model=MetricsResponse,
    summary="获取评估指标",
    description="从 OSS 读取指定数据集的 metrics.json，返回训练完成后的评估指标",
    tags=["结果查询"]
)
def get_metrics(
    dataset: str,
    model: str = "theta",
    current_user: User = Depends(get_current_user),
):
    """
    读取 OSS 上的 metrics.json 文件
    支持两种路径结构:
    - 单模型: results/{username}/{dataset}/{model}/{run_id}/evaluation/metrics.json
    - 多模型: results/{username}/{dataset}/{run_id}/{model}/evaluation/metrics.json
    """
    import oss2
    bucket = get_oss_bucket()

    # 使用通用路径查找函数
    result_path = find_result_path(bucket, current_user.username, dataset, model)

    if not result_path:
        raise HTTPException(status_code=404, detail=f"未找到数据集 {dataset} 模型 {model} 的训练结果")

    # 读取 metrics.json
    obj_key = f"{result_path}/evaluation/metrics.json"

    try:
        result = bucket.get_object(obj_key)
        content = result.read().decode('utf-8')
        import json
        metrics = json.loads(content)
    except oss2.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="metrics.json 文件不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取 metrics.json 失败: {str(e)}")

    return MetricsResponse(
        dataset=dataset,
        model=model,
        metrics=metrics
    )


@app.get(
    "/api/results/{dataset}/visualizations",
    response_model=VisualizationResponse,
    summary="获取可视化文件列表",
    description="列出指定数据集的所有可视化图表文件及其下载链接",
    tags=["结果查询"]
)
def get_visualizations(
    dataset: str,
    model: str = "theta",
    current_user: User = Depends(get_current_user),
):
    """
    获取可视化文件列表
    支持两种路径结构:
    - 单模型: results/{username}/{dataset}/{model}/{run_id}/visualization/zh/
    - 多模型: results/{username}/{dataset}/{run_id}/{model}/visualization/zh/
    """
    import oss2
    from utils.oss_util import OSS_ENDPOINT
    bucket = get_oss_bucket()

    # 使用通用路径查找函数
    result_path = find_result_path(bucket, current_user.username, dataset, model)

    if not result_path:
        raise HTTPException(status_code=404, detail=f"未找到数据集 {dataset} 模型 {model} 的训练结果")

    base_prefix = f"{result_path}/visualization/"

    global_files = []
    topic_files: Dict[str, List] = {}

    # 遍历所有文件
    for obj in oss2.ObjectIterator(bucket, prefix=base_prefix):
        key = obj.key
        if key.endswith('/'):
            continue

        # 提取相对路径（跳过 zh/ 前缀）
        relative_path = key[len(base_prefix):]
        if relative_path.startswith('zh/'):
            relative_path = relative_path[3:]

        if relative_path.startswith('global/'):
            filename = relative_path.split('/')[-1]
            if filename:
                file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
                url = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{key}"
                global_files.append(VisualizationFile(
                    name=filename,
                    path=f"global/{filename}",
                    url=url,
                    size=obj.size,
                    type="global"
                ))
        elif relative_path.startswith('topic/'):
            # topic/topic_N/filename.png
            parts = relative_path.split('/')
            if len(parts) >= 3:
                topic_id = parts[1].replace('topic_', '')
                filename = parts[-1]
                if filename:
                    url = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{key}"
                    if topic_id not in topic_files:
                        topic_files[topic_id] = []
                    topic_files[topic_id].append(VisualizationFile(
                        name=filename,
                        path=f"topic/{parts[1]}/{filename}",
                        url=url,
                        size=obj.size,
                        type="topic"
                    ))

    # 按 topic_id 排序
    topic_files = dict(sorted(topic_files.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0))

    return VisualizationResponse(
        dataset=dataset,
        model=model,
        global_files=global_files,
        topic_files=topic_files
    )


@app.get(
    "/api/results/{dataset}/visualizations/file",
    summary="获取可视化文件内容",
    description="通过后端代理获取可视化文件内容，支持 HTML 和 CSV 文件预览",
    tags=["结果查询"]
)
def get_visualization_file(
    dataset: str,
    path: str,
    model: str = "theta",
    current_user: User = Depends(get_current_user),
):
    """
    获取可视化文件内容（用于预览）
    path: global/xxx.html 或 topic/topic_1/xxx.png
    支持两种路径结构:
    - 单模型: results/{username}/{dataset}/{model}/{run_id}/visualization/zh/{path}
    - 多模型: results/{username}/{dataset}/{run_id}/{model}/visualization/zh/{path}
    """
    import oss2
    from urllib.parse import unquote
    from starlette.responses import Response
    bucket = get_oss_bucket()

    # URL decode path (frontend sends encoded paths)
    path = unquote(path)

    # 使用通用路径查找函数
    result_path = find_result_path(bucket, current_user.username, dataset, model)

    if not result_path:
        raise HTTPException(status_code=404, detail=f"未找到数据集 {dataset} 模型 {model} 的训练结果")

    # 尝试两种路径结构：
    # 1. 有 zh/ 前缀: visualization/zh/{path} (etm 等模型)
    # 2. 无 zh/ 前缀: visualization/{path} (theta 等模型)
    obj_key_zh = f"{result_path}/visualization/zh/{path}"
    obj_key_no_zh = f"{result_path}/visualization/{path}"

    try:
        # 先尝试带 zh/ 的路径
        result = bucket.get_object(obj_key_zh)
        content = result.read()
        content_type = path.split('.')[-1].lower()
    except oss2.exceptions.NoSuchKey:
        # 如果找不到，尝试不带 zh/ 的路径
        try:
            result = bucket.get_object(obj_key_no_zh)
            content = result.read()
            content_type = path.split('.')[-1].lower()
        except oss2.exceptions.NoSuchKey:
            raise HTTPException(status_code=404, detail="文件不存在")

    if content_type == 'html':
        media_type = 'text/html'
    elif content_type == 'csv':
        media_type = 'text/csv'
    elif content_type == 'png':
        media_type = 'image/png'
    elif content_type == 'jpg' or content_type == 'jpeg':
        media_type = 'image/jpeg'
    else:
        media_type = 'application/octet-stream'

    return Response(content=content, media_type=media_type)


@app.delete(
    "/api/datasets/{dataset}",
    summary="删除数据集",
    description="从数据库和 OSS 中删除指定数据集的所有文件",
    tags=["数据集管理"]
)
def delete_dataset(
    dataset: str,
    current_user: User = Depends(get_current_user),
):
    """
    删除数据集：同时删除数据库记录和 OSS 上的文件
    OSS 路径格式: raw_data/{username}/{dataset}/ 和 results/{username}/{dataset}/
    """
    import oss2
    bucket = get_oss_bucket()

    # 1. 删除 OSS 上的 raw_data 目录（批量删除）
    raw_prefix = f"raw_data/{current_user.username}/{dataset}/"
    try:
        raw_keys = [obj.key for obj in oss2.ObjectIterator(bucket, prefix=raw_prefix)]
        if raw_keys:
            bucket.batch_delete_objects(raw_keys)
    except Exception as e:
        print(f"[WARN] 删除 raw_data 失败: {e}")

    # 2. 删除 OSS 上的 results 目录（批量删除）
    results_prefix = f"results/{current_user.username}/{dataset}/"
    try:
        results_keys = [obj.key for obj in oss2.ObjectIterator(bucket, prefix=results_prefix)]
        if results_keys:
            bucket.batch_delete_objects(results_keys)
    except Exception as e:
        print(f"[WARN] 删除 results 失败: {e}")

    # 3. 删除数据库中的相关记录
    db = SessionLocal()
    try:
        # 删除关联的文件记录
        files = db.query(File).filter(
            File.owner_id == current_user.id,
            File.file_path.like(f"raw_data/{current_user.username}/{dataset}/%")
        ).all()
        for f in files:
            db.delete(f)

        # 删除关联的训练任务
        file_ids = [f.id for f in files]
        if file_ids:
            jobs = db.query(TrainingJob).filter(TrainingJob.file_id.in_(file_ids)).all()
            for job in jobs:
                db.delete(job)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"删除数据库记录失败: {e}")
    finally:
        db.close()

    return {"success": True, "message": f"数据集 {dataset} 已删除"}


@app.get(
    "/api/results/{dataset}/visualizations/image",
    summary="获取可视化图片内容",
    description="通过后端代理获取可视化图片内容，解决 CORS 问题",
    tags=["结果查询"]
)
def get_visualization_image(
    dataset: str,
    path: str,
    model: str = "theta",
    current_user: User = Depends(get_current_user),
):
    """
    获取可视化图片内容（用于 AI 分析）
    path: global/xxx.png 或 topic/topic_1/xxx.png
    """
    import oss2
    from urllib.parse import unquote
    from starlette.responses import Response
    bucket = get_oss_bucket()

    # URL decode path (frontend sends encoded paths)
    path = unquote(path)

    # 使用通用路径查找函数
    result_path = find_result_path(bucket, current_user.username, dataset, model)

    if not result_path:
        raise HTTPException(status_code=404, detail=f"未找到数据集 {dataset} 模型 {model} 的训练结果")

    # 尝试两种路径结构：
    # 1. 有 zh/ 前缀: visualization/zh/{path} (etm 等模型)
    # 2. 无 zh/ 前缀: visualization/{path} (theta 等模型)
    obj_key_zh = f"{result_path}/visualization/zh/{path}"
    obj_key_no_zh = f"{result_path}/visualization/{path}"

    try:
        result = bucket.get_object(obj_key_zh)
        content = result.read()
    except oss2.exceptions.NoSuchKey:
        try:
            result = bucket.get_object(obj_key_no_zh)
            content = result.read()
        except oss2.exceptions.NoSuchKey:
            raise HTTPException(status_code=404, detail="文件不存在")

    content_type = path.split('.')[-1].lower()
    if content_type == 'png':
        media_type = 'image/png'
    elif content_type == 'jpg' or content_type == 'jpeg':
        media_type = 'image/jpeg'
    elif content_type == 'gif':
        media_type = 'image/gif'
    elif content_type == 'webp':
        media_type = 'image/webp'
    else:
        media_type = 'application/octet-stream'
    return Response(content=content, media_type=media_type)


# ==================== 结果解读（Agent API 占位符） ====================

class InterpretRequest(BaseModel):
    job_id: Optional[str] = None
    language: Optional[str] = 'zh'
    use_llm: Optional[bool] = True


class InterpretResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@app.post("/api/interpret/metrics", response_model=InterpretResponse, tags=["结果解读"])
def interpret_metrics(
    request: InterpretRequest,
    current_user: User = Depends(get_current_user),
):
    """解读评估指标（占位符接口）"""
    return InterpretResponse(
        success=True,
        message="指标解读功能暂未实现",
        data={"job_id": request.job_id, "interpretation": None}
    )


@app.post("/api/interpret/topics", response_model=InterpretResponse, tags=["结果解读"])
def interpret_topics(
    request: InterpretRequest,
    current_user: User = Depends(get_current_user),
):
    """解读主题语义（占位符接口）"""
    return InterpretResponse(
        success=True,
        message="主题解读功能暂未实现",
        data={"job_id": request.job_id, "interpretation": None}
    )


@app.post("/api/interpret/summary", response_model=InterpretResponse, tags=["结果解读"])
def generate_summary(
    request: InterpretRequest,
    current_user: User = Depends(get_current_user),
):
    """生成分析摘要（占位符接口）"""
    return InterpretResponse(
        success=True,
        message="摘要生成功能暂未实现",
        data={"job_id": request.job_id, "summary": None}
    )


@app.post("/api/vision/analyze-chart", response_model=InterpretResponse, tags=["结果解读"])
def analyze_chart(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_user),
):
    """分析图表（使用 Qwen-VL 自动解读图表内容）"""
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        return InterpretResponse(
            success=False,
            message="DashScope API key not configured",
            data={"analysis": None}
        )

    job_id = request.get("job_id")
    chart_name = request.get("chart_name", "unknown chart")
    analysis_type = request.get("analysis_type", "general")
    language = request.get("language", "zh")

    # 获取图表图片 URL
    # 图表已经上传到 OSS，可以通过公开 URL 访问
    chart_url = request.get("chart_url", "")
    if not chart_url:
        # 前端没有直接传 URL，尝试自己构造（兼容旧版本）
        dataset = request.get("dataset", "")
        if dataset and OSS_ENDPOINT and OSS_BUCKET:
            chart_url = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{dataset}/visualizations/{chart_name}"
        else:
            chart_url = f"{API_BASE_URL}/api/results/{dataset}/visualizations/file?path={chart_name}"

    if not chart_url:
        return InterpretResponse(
            success=False,
            message="无法获取图表URL",
            data={"analysis": None}
        )

    try:
        import dashscope
        import requests
        dashscope.api_key = dashscope_api_key

        # 构建分析提示词
        if language == "zh":
            system_prompt = "你是一位专业的数据分析师，请解读这张主题建模结果图表。用简洁的语言（1-3句话）总结图表展示的内容、主要发现，并给出简要分析。"
            if analysis_type == "loss":
                system_prompt = "这是训练损失曲线图，请分析曲线走势，判断训练是否收敛，并给出结论。用1-2句话回答。"
            elif analysis_type == "topic-word":
                system_prompt = "这是主题词云图，请描述这一主题的主要内容和核心关键词。用1-2句话回答。"
            elif analysis_type == "cluster":
                system_prompt = "这是文档聚类可视化图，请分析聚类结果是否清晰，文档分布有什么特点。用1-3句话回答。"
        else:
            system_prompt = "You are a professional data analyst. Interpret this topic modeling chart in 1-3 concise sentences."

        # 后端下载图片，然后编码传给 Dashscope（解决 OSS 权限/网络问题导致 Dashscope 无法下载）
        response = requests.get(chart_url, timeout=30)
        if response.status_code != 200:
            return InterpretResponse(
                success=False,
                message=f"无法下载图片: HTTP {response.status_code}",
                data={"analysis": f"无法下载图片: HTTP {response.status_code}"}
            )
        # Dashscope 需要 base64 编码的字符串，且必须带有 data URI 前缀
        import base64
        # 根据扩展名判断 mime type
        content_type = chart_name.split('.')[-1].lower()
        mime_type = f"image/{content_type}"
        if content_type == 'jpg':
            mime_type = "image/jpeg"
        image_b64 = base64.b64encode(response.content).decode('utf-8')
        image_data_uri = f"data:{mime_type};base64,{image_b64}"
        content_parts = [
            {"image": image_data_uri},
            {"text": system_prompt}
        ]

        response = dashscope.MultiModalConversation.call(
            model=DASHSCOPE_VL_MODEL,
            messages=[{
                "role": "user",
                "content": content_parts
            }]
        )

        if response.status_code == 200:
            choices = response.output.get("choices", [])
            if choices:
                msg_content = choices[0].get("message", {}).get("content", [])
                if isinstance(msg_content, list) and len(msg_content) > 0:
                    analysis = msg_content[0].get("text", "")
                else:
                    analysis = str(msg_content) if msg_content else ""
            else:
                analysis = ""

            if not analysis:
                analysis = "AI 无法生成有效解读"

            return InterpretResponse(
                success=True,
                message="分析完成",
                data={"analysis": analysis.strip()}
            )
        else:
            # API 返回错误，将错误消息放入 data 让前端显示
            error_msg = f"API错误: {response.message}"
            return InterpretResponse(
                success=False,
                message=error_msg,
                data={"analysis": error_msg}
            )

    except Exception as e:
        # 异常，将错误消息放入 data 让前端显示
        error_msg = f"分析失败: {str(e)}"
        return InterpretResponse(
            success=False,
            message=error_msg,
            data={"analysis": error_msg}
        )
