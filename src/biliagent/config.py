from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# 加载 .env 文件
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BiliSettings(BaseSettings):
    """B站相关配置"""
    sessdata: str = Field(alias="BILI_SESSDATA")
    bili_jct: str = Field(alias="BILI_BILI_JCT")
    buvid3: str = Field(alias="BILI_BUVID3")
    account_uid: int = Field(alias="BILI_ACCOUNT_UID")


class LLMSettings(BaseSettings):
    """LLM 相关配置"""
    api_key: str = Field(alias="LLM_API_KEY")
    base_url: str = Field(default="https://api.moonshot.ai/v1", alias="LLM_BASE_URL")
    model: str = Field(default="kimi-k2.5", alias="LLM_MODEL")


class AppSettings(BaseSettings):
    """应用运行配置"""
    monitor_interval: int = Field(default=60, alias="MONITOR_INTERVAL")
    summary_max_length: int = Field(default=500, alias="SUMMARY_MAX_LENGTH")
    comment_send_interval: int = Field(default=30, alias="COMMENT_SEND_INTERVAL")
    subtitle_max_length: int = Field(default=15000, alias="SUBTITLE_MAX_LENGTH")
    follower_check_enabled: bool = Field(default=True, alias="FOLLOWER_CHECK_ENABLED")
    not_follower_reply: str = Field(
        default="喂喂！连个关注都不点就想使唤我？[生气] 抓到一只企图白嫖野生总结的B友！快乖乖点上关注，不然本课代表要罢工啦！(〃＞目＜)",
        alias="NOT_FOLLOWER_REPLY",
    )
    verify_max_length: int = Field(default=500, alias="VERIFY_MAX_LENGTH")
    database_url: str = Field(
        default="sqlite+aiosqlite:////app/data/biliagent.db",
        alias="DATABASE_URL",
    )


class Settings(BaseSettings):
    """全局配置聚合"""
    bili: BiliSettings = BiliSettings()  # type: ignore[call-arg]
    llm: LLMSettings = LLMSettings()  # type: ignore[call-arg]
    app: AppSettings = AppSettings()

    # 每个 Agent 可独立配置 LLM，默认继承全局
    agent_llm: dict[str, dict[str, str]] = Field(default_factory=dict)

    def get_agent_llm(self, agent_name: str) -> dict[str, str]:
        """获取指定 Agent 的 LLM 配置，未单独配置则使用全局默认"""
        return self.agent_llm.get(agent_name, {
            "model": self.llm.model,
            "base_url": self.llm.base_url,
            "api_key": self.llm.api_key,
        })


settings = Settings()
