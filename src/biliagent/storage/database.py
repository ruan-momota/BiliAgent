import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from biliagent.config import settings


# ---- SQLAlchemy 基类 ----
class Base(DeclarativeBase):
    pass


# ---- 任务记录表 ----
class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String, default="bilibili")
    mention_id: Mapped[str] = mapped_column(String, unique=True)
    video_id: Mapped[str] = mapped_column(String)
    user_id: Mapped[str] = mapped_column(String)
    user_name: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending/processing/completed/failed
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # 关联
    comments: Mapped[list["Comment"]] = relationship(back_populates="task")
    traces: Mapped[list["AgentTrace"]] = relationship(back_populates="task")


# ---- 摘要缓存表 ----
class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String, default="bilibili")
    video_id: Mapped[str] = mapped_column(String)
    video_title: Mapped[str | None] = mapped_column(String, nullable=True)
    summary_text: Mapped[str] = mapped_column(Text)
    has_subtitles: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    __table_args__ = (
        # 同平台同视频唯一
        {"sqlite_autoincrement": True},
    )


# ---- 评论记录表 ----
class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(Integer, ForeignKey("tasks.id"))
    platform: Mapped[str] = mapped_column(String, default="bilibili")
    comment_id: Mapped[str | None] = mapped_column(String, nullable=True)
    content: Mapped[str] = mapped_column(Text)
    floor_number: Mapped[int] = mapped_column(Integer, default=1)
    posted_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    task: Mapped["Task"] = relationship(back_populates="comments")


# ---- Agent 执行追溯表 ----
class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(Integer, ForeignKey("tasks.id"))
    agent_name: Mapped[str] = mapped_column(String)  # supervisor/analyzer/summarizer/reply
    input_data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    output_data: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, default="success")  # success/failed/skipped
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )

    task: Mapped["Task"] = relationship(back_populates="traces")


# ---- 数据库引擎与会话 ----
engine = create_async_engine(settings.app.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """建表（如果不存在）"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
