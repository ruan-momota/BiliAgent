# BiliAgent

基于多 Agent 协同的 B 站视频自动总结机器人。

当其他用户在 B 站视频下 @指定账号时，BiliAgent 自动提取视频字幕、生成摘要，并以评论方式回复。

## 功能特性

- **多 Agent 工作流**：Supervisor → Analyzer → Summarizer → Reply，由 LangGraph 编排
- **LLM 可切换**：每个 Agent 的 LLM 可独立配置（默认 Kimi 2.5）
- **摘要缓存**：同一视频只总结一次，重复 @直接返回缓存结果
- **盖楼策略**：超长评论自动拆分为多楼层回复
- **Cookie 过期检测**：自动检测 B 站 Cookie 状态，Dashboard 实时告警
- **LLM 重试机制**：网络超时、API 限流等临时故障自动重试（最多 3 次，指数退避）
- **可视化 Dashboard**：Next.js 监控面板，查看任务列表、Agent 执行追溯、摘要管理
- **平台可扩展**：抽象平台接口，未来可接入小红书、抖音、YouTube

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.12, FastAPI, Uvicorn |
| Agent 框架 | LangChain + LangGraph |
| LLM | Moonshot AI (Kimi 2.5) via OpenAI 兼容 API |
| B 站 API | bilibili-api-python |
| 数据库 | SQLite (SQLAlchemy + aiosqlite) |
| 前端 | Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui, SWR |
| 包管理 | uv |

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd BiliAgent

# 安装 Python 依赖（需要 uv）
uv sync

# 安装前端依赖
cd dashboard && npm install && cd ..
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入以下信息：

- **B 站 Cookie**：`BILI_SESSDATA`、`BILI_BILI_JCT`、`BILI_BUVID3`、`BILI_ACCOUNT_UID`
  - 登录 B 站后从浏览器 DevTools → Application → Cookies 获取
- **LLM API Key**：`LLM_API_KEY`
  - 从 [platform.moonshot.cn](https://platform.moonshot.cn) 获取 Moonshot API Key

### 3. 启动后端

```bash
uv run uvicorn biliagent.main:app --reload
```

后端启动后：
- API: http://127.0.0.1:8000
- 健康检查: http://127.0.0.1:8000/health
- API 文档: http://127.0.0.1:8000/docs

### 4. 启动前端 Dashboard

```bash
cd dashboard
npm run dev
```

Dashboard: http://localhost:3000

## 运行测试

```bash
uv run pytest tests/ -v
```

## 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BILI_SESSDATA` | B 站 Cookie sessdata | - |
| `BILI_BILI_JCT` | B 站 Cookie bili_jct | - |
| `BILI_BUVID3` | B 站 Cookie buvid3 | - |
| `BILI_ACCOUNT_UID` | B 站账号 UID | - |
| `LLM_API_KEY` | LLM API Key | - |
| `LLM_BASE_URL` | LLM API 地址 | `https://api.moonshot.cn/v1` |
| `LLM_MODEL` | LLM 模型名 | `kimi-k2.5` |
| `MONITOR_INTERVAL` | @消息轮询间隔（秒） | `60` |
| `SUMMARY_MAX_LENGTH` | 摘要最大字数 | `500` |
| `COMMENT_SEND_INTERVAL` | 评论发送间隔（秒） | `30` |
| `SUBTITLE_MAX_LENGTH` | 字幕截断长度 | `15000` |
| `DATABASE_URL` | 数据库连接 | `sqlite+aiosqlite:///./biliagent.db` |

## License

MIT
