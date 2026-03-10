# BiliAgent

基于多 Agent 协同的 B 站视频自动总结机器人。

当其他用户在 B 站视频下 @指定账号时，BiliAgent 自动提取视频字幕、生成摘要，并以评论方式回复。

## 功能特性

- **多 Agent 工作流**：Supervisor → Analyzer → Summarizer → Reply，由 LangGraph 编排
- **LLM 可切换**：每个 Agent 的 LLM 可独立配置（默认 Kimi 2.5 / moonshot-v1-32k）
- **摘要缓存**：同一视频只总结一次，重复 @直接返回缓存结果
- **盖楼策略**：超长评论自动拆分为多楼层回复
- **无字幕处理**：无字幕视频会回复无法总结的原因
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

### 5. 开发测试

无需等待真实 @消息，使用手动触发接口测试：

```bash
curl -X POST http://127.0.0.1:8000/api/test/trigger \
  -H "Content-Type: application/json" \
  -d '{"video_id": "BV1xx411c7XW", "content": "帮我总结一下", "user_name": "test"}'
```

## 项目结构

```
BiliAgent/
├── src/biliagent/
│   ├── main.py              # FastAPI 入口 + Monitor 启动
│   ├── config.py            # 环境变量配置
│   ├── agents/              # 4 个 Agent 实现
│   │   ├── supervisor.py    # 意图判断 + 缓存查询
│   │   ├── analyzer.py      # 视频信息 + 字幕 + 可总结性评估
│   │   ├── summarizer.py    # 摘要生成
│   │   └── reply.py         # 评论格式化 + 发布 + 盖楼
│   ├── graph/               # LangGraph 工作流
│   │   ├── state.py         # AgentState 状态定义
│   │   └── workflow.py      # 4 节点 StateGraph 编排
│   ├── platforms/           # 平台抽象层
│   │   ├── base.py          # PlatformBase 抽象类
│   │   └── bilibili/        # B 站实现
│   ├── prompts/             # 提示词模板
│   ├── models/schemas.py    # Pydantic 数据模型
│   ├── storage/             # 数据库 + 缓存
│   └── api/routes.py        # REST API 端点
├── dashboard/               # Next.js 前端
├── tests/                   # 单元测试 + 集成测试
├── .env.example             # 环境变量模板
└── roadmap.md               # 项目路线图
```

## Agent 工作流

```
@消息 → Supervisor（查缓存/判意图）
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
  缓存命中  Analyzer   忽略
    │       （分析视频）
    │         │
    │    ┌────┴────┐
    │    ▼         ▼
    │  Summarizer  无字幕
    │  （生成摘要）  │
    ▼    ▼         ▼
      Reply Agent
    （格式化+发评论）
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（含 Cookie 状态） |
| GET | `/api/tasks` | 任务列表（支持分页、状态筛选） |
| GET | `/api/tasks/{id}` | 任务详情（含 Agent trace） |
| GET | `/api/stats` | 统计概览 |
| GET | `/api/summaries` | 摘要缓存列表 |
| DELETE | `/api/summaries/{id}` | 删除摘要缓存 |
| POST | `/api/test/trigger` | 手动触发工作流 |

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
| `LLM_MODEL` | LLM 模型名 | `moonshot-v1-32k` |
| `MONITOR_INTERVAL` | @消息轮询间隔（秒） | `60` |
| `SUMMARY_MAX_LENGTH` | 摘要最大字数 | `500` |
| `COMMENT_SEND_INTERVAL` | 评论发送间隔（秒） | `30` |
| `SUBTITLE_MAX_LENGTH` | 字幕截断长度 | `15000` |
| `DATABASE_URL` | 数据库连接 | `sqlite+aiosqlite:///./biliagent.db` |

## License

MIT
