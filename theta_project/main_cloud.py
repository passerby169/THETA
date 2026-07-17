import csv
import io
import os
import base64
import mimetypes
import json
import re
import secrets
import smtplib
import uuid
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
import requests
from sqlalchemy.orm import Session
from starlette.responses import Response, StreamingResponse

from app.database import Base, ChatMessage, File, SessionLocal, TrainingJob, User, VerificationCode, engine, get_db
from services.gpu_provider import submit_training_job
from utils import object_storage as storage
from utils.prompts import AI_CHAT_SYSTEM_PROMPT, AI_CHAT_SYSTEM_PROMPT_MULTI, DASHSCOPE_MODEL, build_chart_analysis_messages

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
KNOWN_MODELS = {"theta", "nvdm", "bertopic", "lda", "hdp", "stm", "btm", "ctm", "etm", "dtm", "prodlda", "gsm"}

Base.metadata.create_all(bind=engine)

app = FastAPI(title="THETA Cloud API", version="2.0.0", docs_url="/docs", redoc_url="/redoc")
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    code: str


class SendVerificationCodeRequest(BaseModel):
    email: EmailStr
    type: str = "register"


class VerifyCodeRequest(BaseModel):
    email: EmailStr
    code: str
    type: str = "register"


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
    dataset_name: Optional[str] = None

    class Config:
        from_attributes = True


class DatasetPreviewResponse(BaseModel):
    columns: List[str]
    rows: List[List[str]]


class UploadTokenResponse(BaseModel):
    credentials: dict = {}
    upload_path: str
    bucket: str
    endpoint: str
    region: str
    provider: str = "r2"
    object_key: Optional[str] = None
    upload_url: Optional[str] = None
    method: str = "PUT"
    headers: dict = {}
    public_url: Optional[str] = None


class UploadCompleteRequest(BaseModel):
    dataset_name: str
    filename: str
    oss_path: Optional[str] = None
    object_key: Optional[str] = None
    file_size: Optional[int] = None


class TrainStartRequest(BaseModel):
    file_id: int
    dataset_name: Optional[str] = None
    model_type: str = "theta"
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
    embedding_provider: Optional[str] = None
    embedding_cloud_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_api_base: Optional[str] = None
    embedding_api_key_env: Optional[str] = None
    embedding_dimensions: Optional[int] = None


class TrainingJobResponse(BaseModel):
    id: int
    user_id: int
    status: str
    dlc_job_id: Optional[str]
    run_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
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


class TrainingStatusResponse(BaseModel):
    job_id: int
    status: str
    dlc_job_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    message: str


class TrainingMetricsResponse(BaseModel):
    job_id: int
    status: str
    metrics: Dict[str, Any] = {}
    epochs: List[int]
    loss: List[float]
    accuracy: List[float]


class TrainingSummaryResponse(BaseModel):
    job_id: int
    summary: Dict[str, Any]


class TrainingCallbackRequest(BaseModel):
    job_id: int
    status: str
    run_id: Optional[str] = None
    error_message: Optional[str] = None
    secret_key: Optional[str] = None


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
    config: Optional[dict] = None


class TopicWordsResponse(BaseModel):
    dataset: str
    model: str
    topics: Dict[str, Any]


class MetricsResponse(BaseModel):
    dataset: str
    model: str
    metrics: Dict[str, Any]


class VisualizationFile(BaseModel):
    name: str
    path: str
    url: str
    size: Optional[int] = 0
    type: str


class VisualizationResponse(BaseModel):
    dataset: str
    model: str
    global_files: List[VisualizationFile]
    topic_files: Dict[str, List[VisualizationFile]]


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[dict] = None
    images: Optional[list] = None
    files: Optional[list] = None


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def verification_secret(email: str, purpose: str, code: str) -> str:
    return f"{normalize_email(email)}:{purpose}:{code}"


def send_verification_email(email: str, code: str) -> bool:
    host = os.getenv("SMTP_HOST")
    if not host:
        print(f"[auth] SMTP is not configured; verification code for {email}: {code}")
        return False

    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_FROM") or username
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes", "on"}

    if not sender:
        print(f"[auth] SMTP_FROM or SMTP_USERNAME is required; verification code for {email}: {code}")
        return False

    message = EmailMessage()
    message["Subject"] = "THETA 注册验证码"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        f"您的 THETA 注册验证码是：{code}\n\n"
        f"验证码 {os.getenv('VERIFICATION_CODE_EXPIRE_MINUTES', '10')} 分钟内有效。"
    )

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            if use_tls:
                server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(message)
        return True
    except Exception as exc:
        print(f"[auth] failed to send verification email to {email}: {exc}")
        print(f"[auth] verification code for {email}: {code}")
        return False


def get_latest_verification_code(db: Session, email: str, purpose: str) -> Optional[VerificationCode]:
    return (
        db.query(VerificationCode)
        .filter(
            VerificationCode.email == normalize_email(email),
            VerificationCode.purpose == purpose,
            VerificationCode.consumed_at.is_(None),
        )
        .order_by(VerificationCode.created_at.desc())
        .first()
    )


def validate_verification_code(db: Session, email: str, purpose: str, code: str) -> VerificationCode:
    record = get_latest_verification_code(db, email, purpose)
    if not record:
        raise HTTPException(status_code=400, detail="验证码不存在或已过期，请重新获取")
    if record.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="验证码已过期，请重新获取")
    if not verify_password(verification_secret(email, purpose, code), record.code_hash):
        raise HTTPException(status_code=400, detail="验证码错误")
    return record


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise error
    except JWTError:
        raise error
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise error
    return user


def file_to_response(file: File) -> FileResponse:
    dataset_name = None
    parts = (file.file_path or "").split("/")
    if len(parts) >= 3 and parts[0] == "raw_data":
        dataset_name = parts[2]
    return FileResponse(
        id=file.id,
        filename=file.filename,
        file_path=file.file_path,
        file_type=file.file_type,
        created_at=file.created_at,
        dataset_name=dataset_name,
    )


def result_path_key(username: str, dataset: str, model: str) -> str:
    prefix1 = f"results/{username}/{dataset}/{model}/"
    run_ids1 = [
        entry.key[len(prefix1):].strip("/")
        for entry in storage.list_objects(prefix1, delimiter="/")
        if entry.is_prefix
    ]
    run_ids1 = [run_id for run_id in run_ids1 if run_id and not run_id.startswith(model)]
    if run_ids1:
        return f"results/{username}/{dataset}/{model}/{sorted(run_ids1)[-1]}"

    prefix2 = f"results/{username}/{dataset}/"
    run_ids2: list[str] = []
    for entry in storage.list_objects(prefix2, delimiter="/"):
        if not entry.is_prefix:
            continue
        potential = entry.key[len(prefix2):].strip("/")
        if not potential or potential in KNOWN_MODELS or "," in potential:
            continue
        model_prefix = f"results/{username}/{dataset}/{potential}/{model}/"
        if storage.list_objects(model_prefix, delimiter="/"):
            run_ids2.append(potential)
    if run_ids2:
        return f"results/{username}/{dataset}/{sorted(run_ids2)[-1]}/{model}"
    return ""


def result_objects(result_path: str):
    prefix = f"{result_path.rstrip('/')}/"
    return [
        entry
        for entry in storage.list_objects(prefix)
        if not entry.is_prefix and not entry.key.endswith("/")
    ]


def result_relative_path(key: str, result_path: str) -> str:
    prefix = f"{result_path.rstrip('/')}/"
    if key.startswith(prefix):
        return key[len(prefix):]
    return key


def is_visualization_key(key: str) -> bool:
    lower = key.lower()
    if not lower.endswith((".png", ".jpg", ".jpeg", ".html", ".csv")):
        return False
    return (
        "visualization" in lower
        or "/global/" in lower
        or "/topic/" in lower
    )


def normalized_visualization_path(relative: str) -> str:
    value = relative.replace("\\", "/")
    if value.startswith("visualization/"):
        value = value[len("visualization/"):]
    if value.startswith("zh/zero_shot/"):
        value = value[len("zh/zero_shot/"):]
    elif value.startswith("zh/"):
        value = value[len("zh/"):]
    return value


def find_result_file(result_path: str, filename: str, contains: Optional[str] = None) -> Optional[str]:
    matches = []
    for entry in result_objects(result_path):
        lower = entry.key.lower()
        if not lower.endswith(filename.lower()):
            continue
        if contains and contains.lower() not in lower:
            continue
        matches.append(entry.key)
    if not matches:
        return None
    return sorted(matches, key=lambda key: (0 if "/theta/" in key.lower() else 1, len(key), key))[0]


def parse_topic_table(content: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for index, row in enumerate(csv.DictReader(io.StringIO(content))):
        raw_topic_id = row.get("topic_id") or row.get("topic") or str(index + 1)
        try:
            topic_id = str(int(raw_topic_id) - 1)
        except Exception:
            topic_id = str(index)
        raw_keywords = row.get("keywords") or row.get("关键词") or row.get("words") or ""
        keywords = [kw.strip() for kw in raw_keywords.replace("，", ",").split(",") if kw.strip()]
        parsed[topic_id] = [[kw, 1.0] for kw in keywords[:10]]
    return parsed


@app.get("/health")
def health():
    return {
        "status": "ok",
        "storage": storage.storage_provider(),
        "gpu_provider": os.getenv("GPU_PROVIDER", "queue"),
    }


@app.post("/api/auth/send-code")
def send_auth_verification_code(request: SendVerificationCodeRequest, db: Session = Depends(get_db)):
    purpose = request.type or "register"
    if purpose not in {"register", "reset_password"}:
        raise HTTPException(status_code=400, detail="Unsupported verification code type")

    email = normalize_email(request.email)
    if purpose == "register":
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

    now = datetime.utcnow()
    min_interval_seconds = int(os.getenv("VERIFICATION_CODE_RESEND_SECONDS", "60"))
    hourly_limit = int(os.getenv("VERIFICATION_CODE_HOURLY_LIMIT", "5"))

    latest_code = (
        db.query(VerificationCode)
        .filter(VerificationCode.email == email, VerificationCode.purpose == purpose)
        .order_by(VerificationCode.created_at.desc())
        .first()
    )
    if latest_code and (now - latest_code.created_at).total_seconds() < min_interval_seconds:
        raise HTTPException(
            status_code=429,
            detail=f"验证码发送过于频繁，请 {min_interval_seconds} 秒后再试",
        )

    recent_count = (
        db.query(VerificationCode)
        .filter(
            VerificationCode.email == email,
            VerificationCode.purpose == purpose,
            VerificationCode.created_at >= now - timedelta(hours=1),
        )
        .count()
    )
    if recent_count >= hourly_limit:
        raise HTTPException(
            status_code=429,
            detail=f"验证码发送次数过多，请 1 小时后再试",
        )

    code = f"{secrets.randbelow(1_000_000):06d}"
    expires_minutes = int(os.getenv("VERIFICATION_CODE_EXPIRE_MINUTES", "10"))

    db.query(VerificationCode).filter(
        VerificationCode.email == email,
        VerificationCode.purpose == purpose,
        VerificationCode.consumed_at.is_(None),
    ).update({VerificationCode.consumed_at: now}, synchronize_session=False)

    record = VerificationCode(
        email=email,
        purpose=purpose,
        code_hash=get_password_hash(verification_secret(email, purpose, code)),
        expires_at=now + timedelta(minutes=expires_minutes),
        created_at=now,
    )
    db.add(record)
    db.commit()

    sent = send_verification_email(email, code)
    response = {
        "message": "验证码已发送，请查收邮箱" if sent else "验证码已生成，请查看后端日志或配置 SMTP 邮件服务",
    }
    if os.getenv("AUTH_DEV_EXPOSE_CODE", "").lower() in {"1", "true", "yes", "on"}:
        response["debug_code"] = code
    return response


@app.post("/api/auth/verify-code")
def verify_auth_code(request: VerifyCodeRequest, db: Session = Depends(get_db)):
    purpose = request.type or "register"
    try:
        validate_verification_code(db, request.email, purpose, request.code)
    except HTTPException:
        return {"valid": False}
    return {"valid": True}


@app.post("/api/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    email = normalize_email(user_data.email)
    existing = db.query(User).filter((User.username == user_data.username) | (User.email == email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already registered")

    code_record = validate_verification_code(db, email, "register", user_data.code)
    user = User(
        username=user_data.username,
        email=email,
        hashed_password=get_password_hash(user_data.password),
        is_active=True,
    )
    code_record.consumed_at = datetime.utcnow()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return {"access_token": create_access_token({"sub": user.username, "email": user.email}), "token_type": "bearer"}


@app.get("/api/auth/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user


@app.post("/api/auth/logout")
def logout():
    return {"message": "Logged out successfully"}


@app.get("/api/oss/sts-token", response_model=UploadTokenResponse)
def get_upload_token(
    dataset_name: str,
    filename: str = "upload.bin",
    content_type: str = "application/octet-stream",
    current_user: User = Depends(get_current_user),
):
    key = storage.raw_data_key(current_user.username, dataset_name, filename)
    signed = storage.create_presigned_upload(key, content_type=content_type)
    return UploadTokenResponse(
        credentials={},
        upload_path=signed["upload_path"],
        bucket=signed["bucket"],
        endpoint=signed["endpoint"],
        region=signed["region"],
        provider=signed["provider"],
        object_key=signed["object_key"],
        upload_url=signed["upload_url"],
        method=signed["method"],
        headers=signed["headers"],
        public_url=signed["public_url"],
    )


@app.post("/api/upload/complete", response_model=FileResponse, status_code=status.HTTP_201_CREATED)
def upload_complete(
    request: UploadCompleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    key = request.object_key or request.oss_path or storage.raw_data_key(
        current_user.username, request.dataset_name, request.filename
    )
    existing = db.query(File).filter(File.owner_id == current_user.id, File.file_path == key).first()
    if existing:
        existing.created_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return file_to_response(existing)
    db_file = File(
        owner_id=current_user.id,
        filename=request.filename,
        file_path=key,
        file_type="r2_upload",
        created_at=datetime.utcnow(),
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return file_to_response(db_file)


@app.get("/api/files", response_model=List[FileResponse])
def list_files(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    files = db.query(File).filter(File.owner_id == current_user.id).all()
    return [file_to_response(file) for file in files if (file.file_path or "").startswith("raw_data/")]


@app.get("/api/datasets/{dataset}/preview", response_model=DatasetPreviewResponse)
def preview_dataset(dataset: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prefix = f"raw_data/{current_user.username}/{dataset}/"
    file_record = (
        db.query(File)
        .filter(File.owner_id == current_user.id, File.file_path.like(f"{prefix}%"))
        .filter(File.filename.ilike("%.csv"))
        .order_by(File.created_at.desc())
        .first()
    )

    object_key = file_record.file_path if file_record else None
    if not object_key:
        csv_objects = [
            entry for entry in storage.list_objects(prefix)
            if not entry.is_prefix and entry.key.lower().endswith(".csv")
        ]
        if csv_objects:
            object_key = sorted(csv_objects, key=lambda item: item.last_modified or datetime.min, reverse=True)[0].key

    if not object_key:
        raise HTTPException(status_code=404, detail="No CSV file found for this dataset")

    try:
        content = storage.get_object_bytes(object_key).decode("utf-8-sig")
    except UnicodeDecodeError:
        content = storage.get_object_bytes(object_key).decode("gb18030", errors="replace")

    reader = csv.reader(io.StringIO(content))
    try:
        columns = [str(cell).strip() for cell in next(reader)]
    except StopIteration:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    if not columns or all(not col for col in columns):
        raise HTTPException(status_code=400, detail="CSV header row is empty")

    rows: List[List[str]] = []
    for row in reader:
        rows.append([str(cell) for cell in row])
        if len(rows) >= 5:
            break

    return DatasetPreviewResponse(columns=columns, rows=rows)


@app.get("/api/preprocessing/check/{dataset}", response_model=PreprocessingStatusResponse)
def check_preprocessing_status(dataset: str, current_user: User = Depends(get_current_user)):
    prefix = f"results/{current_user.username}/{dataset}/"
    has_bow = has_embeddings = has_vocab = False
    bow_path = embedding_path = vocab_path = None
    for entry in storage.list_objects(prefix):
        key = entry.key.lower()
        if "bow_matrix" in key:
            has_bow, bow_path = True, entry.key
        elif "embeddings.npy" in key or "embedding.npy" in key:
            has_embeddings, embedding_path = True, entry.key
        elif "vocab.json" in key or "vocab.txt" in key:
            has_vocab, vocab_path = True, entry.key
        if has_bow and has_embeddings and has_vocab:
            break
    return PreprocessingStatusResponse(
        dataset=dataset,
        has_bow=has_bow,
        has_embeddings=has_embeddings,
        ready_for_training=has_bow and has_embeddings and has_vocab,
        bow_path=bow_path,
        embedding_path=embedding_path,
        vocab_path=vocab_path,
    )


@app.post("/api/preprocessing/start", response_model=PreprocessingJobResponse, status_code=status.HTTP_201_CREATED)
def start_preprocessing(request: StartPreprocessingRequest, current_user: User = Depends(get_current_user)):
    now = datetime.utcnow().isoformat()
    return PreprocessingJobResponse(
        job_id=f"prep_{uuid.uuid4().hex[:12]}",
        dataset=request.dataset,
        status="completed",
        progress=100,
        message="Preprocessing is performed inside the GPU training job",
        created_at=now,
        updated_at=now,
    )


@app.get("/api/preprocessing/{job_id}", response_model=PreprocessingJobResponse)
def get_preprocessing_job(job_id: str, current_user: User = Depends(get_current_user)):
    return PreprocessingJobResponse(
        job_id=job_id,
        dataset="",
        status="completed",
        progress=100,
        message="Preprocessing is handled by the GPU job",
    )


@app.post("/api/train/start", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
def start_training(request: TrainStartRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    file_record = db.query(File).filter(File.id == request.file_id, File.owner_id == current_user.id).first()
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found or access denied")
    uploaded_dataset = file_to_response(file_record).dataset_name
    dataset_name = request.dataset_name or uploaded_dataset or f"dataset_{request.file_id}"
    job = TrainingJob(
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
        created_at=datetime.utcnow(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    run_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    payload = {
        "job_id": job.id,
        "run_id": run_id,
        "callback_url": f"{API_BASE_URL.rstrip('/')}/api/train/callback",
        "callback_secret": SECRET_KEY,
        "username": current_user.username,
        "dataset_name": dataset_name,
        "input_key": file_record.file_path,
        "raw_data_prefix": f"raw_data/{current_user.username}/{dataset_name}/",
        "output_prefix": f"results/{current_user.username}/{dataset_name}/{request.model_type}/{run_id}/",
        "model_type": request.model_type,
        "model_size": request.model_size,
        "mode": request.mode,
        "num_topics": request.num_topics,
        "epochs": request.epochs,
        "batch_size": request.batch_size or 64,
        "learning_rate": request.learning_rate or 0.002,
        "hidden_dim": request.hidden_dim or 512,
        "patience": request.patience or 10,
        "vocab_size": request.vocab_size or 5000,
        "language": request.language or "chinese",
        "embedding_provider": request.embedding_provider,
        "embedding_cloud_provider": request.embedding_cloud_provider,
        "embedding_model": request.embedding_model,
        "embedding_api_base": request.embedding_api_base,
        "embedding_api_key_env": request.embedding_api_key_env,
        "embedding_dimensions": request.embedding_dimensions,
        "storage_provider": storage.storage_provider(),
        "storage_bucket": storage.bucket_name(),
    }
    try:
        submitted = submit_training_job(payload)
        job.dlc_job_id = submitted.external_job_id
        job.run_id = submitted.run_id
        job.status = "running" if submitted.status in {"running", "submitted", "queued"} else submitted.status
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
    db.commit()
    db.refresh(job)
    return job


def get_job_for_user(job_id: int, user_id: int, db: Session) -> TrainingJob:
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id, TrainingJob.user_id == user_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@app.get("/api/train/{job_id}/status", response_model=TrainingStatusResponse)
def get_training_status(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = get_job_for_user(job_id, current_user.id, db)
    message = job.error_message or (
        "Training completed" if job.status == "succeeded" else "Training job is running on external training worker"
    )
    return TrainingStatusResponse(
        job_id=job.id,
        status=job.status,
        dlc_job_id=job.dlc_job_id,
        error_message=job.error_message,
        created_at=job.created_at,
        message=message,
    )


@app.get("/api/train/{job_id}/metrics", response_model=TrainingMetricsResponse)
def get_training_metrics(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = get_job_for_user(job_id, current_user.id, db)
    file_record = db.query(File).filter(File.id == job.file_id).first()
    dataset = file_to_response(file_record).dataset_name if file_record else ""
    epochs: list[int] = []
    loss: list[float] = []
    accuracy: list[float] = []
    if dataset and job.run_id:
        key = f"results/{current_user.username}/{dataset}/{job.model_type}/{job.run_id}/training_log.json"
        try:
            data = storage.get_object_json(key)
            for item in data.get("metrics", []):
                epochs.append(int(item.get("epoch", len(epochs) + 1)))
                loss.append(float(item.get("loss", 0.0)))
                accuracy.append(float(item.get("accuracy", 0.0)))
        except Exception:
            pass
    metrics = {
        "last_loss": loss[-1] if loss else None,
        "last_accuracy": accuracy[-1] if accuracy else None,
        "epochs": len(epochs),
    }
    return TrainingMetricsResponse(
        job_id=job.id,
        status=job.status,
        metrics=metrics,
        epochs=epochs,
        loss=loss,
        accuracy=accuracy,
    )


@app.get("/api/train/{job_id}/summary", response_model=TrainingSummaryResponse)
def get_training_summary(job_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = get_job_for_user(job_id, current_user.id, db)
    file_record = db.query(File).filter(File.id == job.file_id).first()
    dataset = file_to_response(file_record).dataset_name if file_record else ""
    top_words: List[List[str]] = []
    if dataset and job.run_id:
        for model in ["theta", job.model_type or "theta"]:
            result_path = result_path_key(current_user.username, dataset, model)
            if not result_path:
                continue
            key = find_result_file(result_path, "topic_words.json")
            if not key:
                continue
            try:
                payload = storage.get_object_json(key)
                if isinstance(payload, dict):
                    topics = payload.get("topics", payload)
                    if isinstance(topics, dict):
                        top_words = [
                            words if isinstance(words, list) else str(words).split()
                            for words in topics.values()
                        ]
                    elif isinstance(topics, list):
                        top_words = topics
                    break
            except Exception:
                continue
    return TrainingSummaryResponse(
        job_id=job.id,
        summary={
            "num_topics": job.num_topics or len(top_words),
            "top_words": top_words,
            "status": job.status,
            "run_id": job.run_id,
            "model_type": job.model_type,
            "epochs": job.epochs,
            "model_size": job.model_size,
            "vocab_size": job.vocab_size,
        },
    )


@app.post("/api/train/callback")
def training_callback(request: dict, db: Session = Depends(get_db)):
    if request.get("secret_key") and request.get("secret_key") != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid secret key")
    job = db.query(TrainingJob).filter(TrainingJob.id == int(request["job_id"])).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    status_value = request.get("status")
    if status_value not in {"running", "succeeded", "failed"}:
        raise HTTPException(status_code=400, detail="Invalid status")
    job.status = status_value
    job.run_id = request.get("run_id") or job.run_id
    job.error_message = request.get("error_message") or job.error_message
    db.commit()
    return {"success": True, "message": f"Job {job.id} updated to {job.status}"}


@app.get("/api/train/jobs", response_model=List[TrainingJobResponse])
def get_training_jobs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return (
        db.query(TrainingJob)
        .filter(TrainingJob.user_id == current_user.id)
        .order_by(TrainingJob.created_at.desc())
        .all()
    )


@app.get("/api/data/oss-datasets")
def list_datasets(current_user: User = Depends(get_current_user)):
    try:
        base = f"results/{current_user.username}/"
        datasets = []
        for entry in storage.list_objects(base, delimiter="/"):
            if not entry.is_prefix:
                continue
            dataset = entry.key[len(base):].strip("/")
            chart_count = sum(
                1
                for obj in storage.list_objects(entry.key)
                if obj.key.lower().endswith((".png", ".jpg", ".jpeg")) and is_visualization_key(obj.key)
            )
            if chart_count > 0:
                datasets.append({"name": dataset, "chart_count": chart_count})
        return {"datasets": sorted(datasets, key=lambda item: item["name"])}
    except Exception as exc:
        print(f"Failed to list datasets: {exc}")
        return {"datasets": []}


@app.get("/api/results/{dataset}/models")
def get_available_models(dataset: str, current_user: User = Depends(get_current_user)):
    prefix = f"results/{current_user.username}/{dataset}/"
    models = set()
    for entry in storage.list_objects(prefix, delimiter="/"):
        if not entry.is_prefix:
            continue
        relative = entry.key[len(prefix):].strip("/")
        if relative in KNOWN_MODELS:
            models.add(relative)
        else:
            for sub in storage.list_objects(entry.key, delimiter="/"):
                sub_relative = sub.key[len(entry.key):].strip("/")
                if sub_relative in KNOWN_MODELS:
                    models.add(sub_relative)
    return {"dataset": dataset, "models": sorted(models)}


@app.get("/api/results/{dataset}/topic-words", response_model=TopicWordsResponse)
def get_topic_words(dataset: str, model: str = "theta", current_user: User = Depends(get_current_user)):
    result_path = result_path_key(current_user.username, dataset, model)
    if not result_path:
        raise HTTPException(status_code=404, detail="Training result not found")
    topics: Any = None
    topic_key = find_result_file(result_path, "topic_words.json")
    if topic_key:
        topics = storage.get_object_json(topic_key)
    if topics is None:
        csv_key = find_result_file(result_path, "主题表.csv")
        if csv_key:
            content = storage.get_object_text(csv_key, "utf-8-sig")
            topics = parse_topic_table(content)
    if topics is None:
        raise HTTPException(status_code=404, detail="Topic words not found")
    if isinstance(topics, list):
        topics = {str(i): value for i, value in enumerate(topics)}
    return TopicWordsResponse(dataset=dataset, model=model, topics=topics)


@app.get("/api/results/{dataset}/metrics", response_model=MetricsResponse)
def get_metrics(dataset: str, model: str = "theta", current_user: User = Depends(get_current_user)):
    result_path = result_path_key(current_user.username, dataset, model)
    if not result_path:
        raise HTTPException(status_code=404, detail="Training result not found")
    metrics_key = None
    for candidate in [
        f"{result_path}/evaluation/metrics.json",
        f"{result_path}/metrics_zero_shot.json",
        f"{result_path}/metrics.json",
    ]:
        if storage.object_exists(candidate):
            metrics_key = candidate
            break
    if not metrics_key:
        for entry in result_objects(result_path):
            name = entry.key.rsplit("/", 1)[-1].lower()
            if name.startswith("metrics") and name.endswith(".json"):
                metrics_key = entry.key
                break
    if not metrics_key:
        raise HTTPException(status_code=404, detail="Metrics not found")
    metrics = storage.get_object_json(metrics_key)
    return MetricsResponse(dataset=dataset, model=model, metrics=metrics)


@app.get("/api/results/{dataset}/visualizations", response_model=VisualizationResponse)
def get_visualizations(dataset: str, model: str = "theta", current_user: User = Depends(get_current_user)):
    result_path = result_path_key(current_user.username, dataset, model)
    if not result_path:
        raise HTTPException(status_code=404, detail="Training result not found")
    global_files: list[VisualizationFile] = []
    topic_files: dict[str, list[VisualizationFile]] = {}
    for entry in result_objects(result_path):
        if entry.is_prefix or entry.key.endswith("/"):
            continue
        if not is_visualization_key(entry.key):
            continue
        relative = normalized_visualization_path(result_relative_path(entry.key, result_path))
        if relative.startswith("global/"):
            filename = relative.split("/")[-1]
            global_files.append(
                VisualizationFile(
                    name=filename,
                    path=f"global/{filename}",
                    url=storage.object_url(entry.key),
                    size=entry.size,
                    type="global",
                )
            )
        elif relative.startswith("topic/"):
            parts = relative.split("/")
            if len(parts) >= 3:
                topic_id = parts[1].replace("topic_", "")
                filename = parts[-1]
                topic_files.setdefault(topic_id, []).append(
                    VisualizationFile(
                        name=filename,
                        path=f"topic/{parts[1]}/{filename}",
                        url=storage.object_url(entry.key),
                        size=entry.size,
                        type="topic",
                    )
                )
    return VisualizationResponse(
        dataset=dataset,
        model=model,
        global_files=global_files,
        topic_files=dict(sorted(topic_files.items())),
    )


def visualization_object_key(username: str, dataset: str, model: str, path: str) -> str:
    result_path = result_path_key(username, dataset, model)
    if not result_path:
        raise HTTPException(status_code=404, detail="Training result not found")
    clean_path = unquote(path).replace("\\", "/").lstrip("/")
    for candidate in [
        f"{result_path}/visualization/zh/{clean_path}",
        f"{result_path}/visualization/{clean_path}",
        f"{result_path}/zh/zero_shot/{clean_path}",
        f"{result_path}/zh/{clean_path}",
    ]:
        if storage.object_exists(candidate):
            return candidate
    for entry in result_objects(result_path):
        if normalized_visualization_path(result_relative_path(entry.key, result_path)).endswith(clean_path):
            return entry.key
    raise HTTPException(status_code=404, detail="Visualization file not found")


@app.get("/api/results/{dataset}/visualizations/file")
def get_visualization_file(dataset: str, path: str, model: str = "theta", current_user: User = Depends(get_current_user)):
    key = visualization_object_key(current_user.username, dataset, model, path)
    content = storage.get_object_bytes(key)
    ext = key.rsplit(".", 1)[-1].lower()
    media = {
        "html": "text/html",
        "csv": "text/csv",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }.get(ext, "application/octet-stream")
    return Response(content=content, media_type=media)


@app.get("/api/results/{dataset}/visualizations/image")
def get_visualization_image(dataset: str, path: str, model: str = "theta", current_user: User = Depends(get_current_user)):
    return get_visualization_file(dataset, path, model, current_user)


@app.delete("/api/datasets/{dataset}")
def delete_dataset(dataset: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    files = db.query(File).filter(
        File.owner_id == current_user.id,
        File.file_path.like(f"raw_data/{current_user.username}/{dataset}/%"),
    ).all()
    file_ids = [file.id for file in files]
    deleted_jobs = 0
    if file_ids:
        jobs = db.query(TrainingJob).filter(
            TrainingJob.user_id == current_user.id,
            TrainingJob.file_id.in_(file_ids),
        ).all()
        deleted_jobs = len(jobs)
        for job in jobs:
            db.delete(job)

    storage.delete_prefix(f"raw_data/{current_user.username}/{dataset}/")
    storage.delete_prefix(f"results/{current_user.username}/{dataset}/")
    for file in files:
        db.delete(file)
    db.commit()
    return {
        "success": True,
        "message": f"Dataset {dataset} deleted",
        "deleted_files": len(files),
        "deleted_training_jobs": deleted_jobs,
    }


def _strip_thinking_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def _summarize_chat_attachments(request: ChatRequest) -> str:
    lines: List[str] = []
    for index, image in enumerate(request.images or [], start=1):
        if not isinstance(image, dict):
            continue
        details = [
            f"图片{index}",
            f"名称={image.get('name')}" if image.get("name") else "",
            f"数据集={image.get('dataset')}" if image.get("dataset") else "",
            f"结果路径={image.get('path')}" if image.get("path") else "",
            f"类型={image.get('mimeType') or image.get('mime_type')}" if image.get("mimeType") or image.get("mime_type") else "",
        ]
        lines.append("；".join(part for part in details if part))
    for index, file in enumerate(request.files or [], start=1):
        if not isinstance(file, dict):
            continue
        details = [
            f"文件{index}",
            f"名称={file.get('name')}" if file.get("name") else "",
            f"类型={file.get('mimeType') or file.get('mime_type')}" if file.get("mimeType") or file.get("mime_type") else "",
        ]
        lines.append("；".join(part for part in details if part))
    return "\n".join(lines)


def _build_chat_user_text(request: ChatRequest) -> str:
    text = request.message or ""
    attachment_summary = _summarize_chat_attachments(request)
    if attachment_summary:
        text = f"{text}\n\n用户随消息附加了以下资料：\n{attachment_summary}".strip()
    return text


def _build_openai_user_content(request: ChatRequest) -> Any:
    text = _build_chat_user_text(request)
    image_parts: List[Dict[str, Any]] = []
    for image in request.images or []:
        if not isinstance(image, dict):
            continue
        image_url = image.get("dataUrl") or image.get("data_url") or image.get("url")
        if not isinstance(image_url, str) or not image_url.strip():
            continue
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": image_url.strip(), "detail": os.getenv("VISION_IMAGE_DETAIL", "low")},
        })

    if not image_parts:
        return text

    return [{"type": "text", "text": text}, *image_parts]


def _chat_with_openai_compatible(request: ChatRequest) -> Optional[str]:
    api_key = (
        os.getenv("CHAT_API_KEY")
        or os.getenv("MINIMAX_API_KEY")
        or os.getenv("VISION_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        return None

    api_base = (
        os.getenv("CHAT_API_BASE")
        or os.getenv("MINIMAX_API_BASE")
        or os.getenv("VISION_API_BASE")
        or os.getenv("EMBEDDING_API_BASE")
        or "https://api.minimaxi.com/v1"
    ).rstrip("/")
    has_images = bool(request.images)
    if has_images:
        model = (
            os.getenv("CHAT_VISION_MODEL")
            or os.getenv("VISION_MODEL")
            or os.getenv("MINIMAX_VISION_MODEL")
            or os.getenv("CHAT_MODEL")
            or os.getenv("MINIMAX_TEXT_MODEL")
            or "MiniMax-M3"
        )
    else:
        model = (
            os.getenv("CHAT_MODEL")
            or os.getenv("MINIMAX_TEXT_MODEL")
            or os.getenv("VISION_MODEL")
            or os.getenv("EMBEDDING_MODEL")
            or "MiniMax-M1"
        )
    system_prompt = AI_CHAT_SYSTEM_PROMPT_MULTI if has_images or request.files else AI_CHAT_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_openai_user_content(request)},
    ]
    if request.context:
        messages.insert(1, {"role": "system", "content": f"当前页面上下文：{json.dumps(request.context, ensure_ascii=False)}"})

    response = requests.post(
        f"{api_base}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": float(os.getenv("CHAT_TEMPERATURE", "0.4")),
            "max_tokens": int(os.getenv("CHAT_MAX_TOKENS", "800")),
        },
        timeout=int(os.getenv("CHAT_TIMEOUT_SECONDS", "45")),
    )
    if response.status_code >= 400:
        raise RuntimeError(f"chat request failed ({response.status_code}): {response.text[:500]}")

    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return _strip_thinking_tags(str(content or ""))


def _chat_with_dashscope(request: ChatRequest) -> Optional[str]:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    import dashscope

    dashscope.api_key = api_key
    response = dashscope.Generation.call(
        model=DASHSCOPE_MODEL,
        messages=[
            {"role": "system", "content": AI_CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": _build_chat_user_text(request)},
        ],
    )
    if response.status_code == 200:
        return response.output.get("text")
    raise RuntimeError(f"DashScope request failed ({response.status_code})")


def _generate_chat_answer(request: ChatRequest) -> str:
    if (
        os.getenv("CHAT_API_KEY")
        or os.getenv("MINIMAX_API_KEY")
        or os.getenv("VISION_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    ):
        try:
            answer = _chat_with_openai_compatible(request)
            if answer:
                return answer
        except Exception as exc:
            return f"AI 请求失败：{exc}"

    try:
        answer = _chat_with_dashscope(request)
        if answer:
            return answer
    except Exception as exc:
        return f"AI 请求失败：{exc}"

    return "AI 聊天未配置：请在后端环境变量设置 CHAT_API_KEY、MINIMAX_API_KEY、VISION_API_KEY 或 DASHSCOPE_API_KEY。图片分析至少需要 VISION_API_KEY 或 MINIMAX_API_KEY。"


@app.post("/api/agent/chat")
@app.post("/api/chat")
def chat(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    answer = _generate_chat_answer(request)
    session_id = request.session_id or "default"
    db.add(ChatMessage(user_id=current_user.id, session_id=session_id, role="user", content=request.message, created_at=datetime.utcnow()))
    db.add(ChatMessage(user_id=current_user.id, session_id=session_id, role="ai", content=answer, created_at=datetime.utcnow()))
    db.commit()
    return {"message": answer, "session_id": session_id, "created_at": datetime.utcnow().isoformat()}


@app.post("/api/agent/chat/stream")
def chat_stream(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = chat(request, current_user, db)

    def event_stream():
        yield f"data: {json.dumps({'type': 'content', 'content': result['message']}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/chat/history/{session_id}")
def get_chat_history(session_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == current_user.id, ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return {
        "session_id": session_id,
        "messages": [
            {"role": message.role, "content": message.content, "created_at": message.created_at.isoformat()}
            for message in messages
        ],
    }


@app.post("/api/chat/history/{session_id}")
def save_chat_message(session_id: str, payload: Dict[str, Any], current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    message = ChatMessage(
        user_id=current_user.id,
        session_id=session_id,
        role=payload.get("role", "user"),
        content=payload.get("content", ""),
        created_at=datetime.utcnow(),
    )
    db.add(message)
    db.commit()
    return {"id": message.id, "role": message.role, "content": message.content, "created_at": message.created_at.isoformat()}


@app.post("/api/interpret/metrics")
@app.post("/api/interpret/topics")
@app.post("/api/interpret/summary")
def interpret_placeholder(payload: Dict[str, Any], current_user: User = Depends(get_current_user)):
    return {"success": True, "message": "Interpretation is queued for a later AI service pass", "data": payload}


def sanitize_vision_analysis(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if cleaned.lower().startswith("<think>"):
        parts = re.split(r"</think>", cleaned, flags=re.IGNORECASE, maxsplit=1)
        cleaned = parts[1].strip() if len(parts) > 1 else ""
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


@app.post("/api/vision/analyze-chart")
def analyze_chart(payload: Dict[str, Any], current_user: User = Depends(get_current_user)):
    provider = os.getenv("VISION_PROVIDER", "minimax").lower()
    if provider != "minimax":
        raise HTTPException(status_code=400, detail=f"Unsupported vision provider: {provider}")

    api_key = os.getenv("VISION_API_KEY") or os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="VISION_API_KEY or MINIMAX_API_KEY is not configured")

    dataset = payload.get("dataset")
    model = payload.get("model") or payload.get("model_name") or "theta"
    chart_path = payload.get("chart_path") or payload.get("path")
    chart_name = payload.get("chart_name") or chart_path or "chart"
    analysis_type = payload.get("analysis_type", "general")
    language = payload.get("language", "zh")

    if not dataset or not chart_path:
        raise HTTPException(status_code=400, detail="dataset and chart_path are required")

    key = visualization_object_key(current_user.username, dataset, model, chart_path)
    ext = key.rsplit(".", 1)[-1].lower()
    if ext not in {"png", "jpg", "jpeg", "webp", "gif"}:
        raise HTTPException(status_code=400, detail="Only image chart files can be analyzed")

    clean_path = unquote(chart_path).replace("\\", "/").lstrip("/")
    cache_version = os.getenv("VISION_ANALYSIS_CACHE_VERSION", "v2")
    cache_key = f"{result_path_key(current_user.username, dataset, model)}/analysis/{cache_version}/{clean_path}.json"
    try:
        cached = storage.get_object_json(cache_key)
        analysis = sanitize_vision_analysis(cached.get("analysis", ""))
        if analysis:
            return {"success": True, "message": "Chart analysis loaded from cache", "data": {"analysis": analysis}}
    except Exception:
        pass

    image_bytes = storage.get_object_bytes(key)
    if len(image_bytes) > int(os.getenv("VISION_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))):
        raise HTTPException(status_code=413, detail="Chart image is too large for vision analysis")

    media_type = mimetypes.guess_type(key)[0] or ("image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}")
    image_data_url = f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    api_base = os.getenv("VISION_API_BASE", "https://api.minimaxi.com/v1").rstrip("/")
    vision_model = os.getenv("VISION_MODEL", "MiniMax-M3")
    max_tokens = max(800, int(os.getenv("VISION_MAX_TOKENS", "1000")))
    detail = os.getenv("VISION_IMAGE_DETAIL", "low")
    messages = build_chart_analysis_messages(
        chart_name=chart_name,
        analysis_type=analysis_type,
        language=language,
        image_data_url=image_data_url,
        image_detail=detail,
    )

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": vision_model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=int(os.getenv("VISION_REQUEST_TIMEOUT", "60")),
        )
        response.raise_for_status()
        result = response.json()
        analysis = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        analysis = sanitize_vision_analysis(analysis)
    except requests.HTTPError as exc:
        detail_text = exc.response.text[:500] if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"MiniMax vision request failed: {detail_text}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MiniMax vision request failed: {exc}") from exc

    if not analysis:
        raise HTTPException(status_code=502, detail="MiniMax vision response did not contain analysis text")

    try:
        storage.put_object_bytes(
            cache_key,
            body=json.dumps(
                {"analysis": analysis, "provider": "minimax", "model": vision_model},
                ensure_ascii=False,
            ).encode("utf-8"),
            content_type="application/json",
        )
    except Exception:
        pass

    return {"success": True, "message": "Chart analysis generated", "data": {"analysis": analysis}}
