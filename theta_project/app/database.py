import os
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Index
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# SQLite needs special handling for multi-threading
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    # PostgreSQL uses default connection parameters
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)

    files = relationship("File", back_populates="owner")


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)

    owner = relationship("User", back_populates="files")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    model_type = Column(String, nullable=True)
    model_size = Column(String, nullable=True)
    num_topics = Column(Integer, nullable=True)
    epochs = Column(Integer, nullable=True)
    batch_size = Column(Integer, nullable=True)
    learning_rate = Column(Float, nullable=True)
    hidden_dim = Column(Integer, nullable=True)
    patience = Column(Integer, nullable=True)
    vocab_size = Column(Integer, nullable=True)
    mode = Column(String, nullable=True)
    language = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending", index=True)
    dlc_job_id = Column(String, nullable=True)
    run_id = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, index=True)

    user = relationship("User")
    file = relationship("File")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("ix_chat_messages_user_session", "user_id", "session_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, nullable=False, default="default")
    role = Column(String, nullable=False)  # "user" or "ai"
    content = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)

    user = relationship("User")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
