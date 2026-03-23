# BiliAgent

基于多 Agent 协同的 B 站视频自动总结机器人。

当其他用户在 B 站视频下 @指定账号时，BiliAgent 自动识别用户意图（总结 / 鉴别 / 问答），提取视频字幕，由对应 Agent 处理后以评论方式回复。

## 功能特性

### 核心功能

- **视频摘要生成**：自动提取字幕，生成结构化视频摘要并发评论回复
- **视频鉴别与评论**：识别「评价/鉴别」意图，通过联网搜索交叉验证视频内容，生成带观点的鉴别评论（赞同/反对/质疑/中立）
- **长视频 Map-Reduce 摘要**：超长字幕（>15000 字符）自动分块 → 并行提取要点 → 合并最终摘要，避免 lost-in-the-middle 问题
- **RAG 基础设施**：字幕分块 + Embedding 存入 ChromaDB，支持向量检索（为问答功能奠定基础）

### 智能特性

- **多 Agent 工作流**：Supervisor → Analyzer → Summarizer / Verifier → Reply，由 LangGraph 编排
- **意图识别**：Supervisor Agent 自动分类用户意图（summarize / verify / ignore），路由到对应处理链路
- **摘要缓存**：同一视频只总结一次，重复 @直接返回缓存
- **鉴别缓存 + 语义匹配**：同视频可有多条不同角度的鉴别记录，由 LLM 判断新提问与历史鉴别的语义相似度，相似则复用缓存
- **关注检查**：仅为已关注用户提供服务，未关注用户收到傲娇人设的求关注提示
- **联网搜索**：Verifier Agent 通过 Kimi `$web_search` 获取外部信息交叉验证视频内容

### 工程特性

- **LLM 可切换**：每个 Agent 的 LLM 可独立配置（默认 Kimi K2.5，128K 上下文）
- **平台可扩展**：抽象平台接口 `PlatformBase`，未来可接入小红书、抖音、YouTube
- **盖楼策略**：超长评论自动拆分为多楼层回复
- **Cookie 过期检测**：启动时自动检测 + 运行中感知，Dashboard 实时告警
- **LLM 重试机制**：网络超时、429 限流、5xx 等临时故障自动重试（最多 3 次，指数退避）
- **Agent 执行追溯**：每个节点的输入/输出/耗时/状态全部记录，Dashboard 可追溯完整生成过程
- **可视化 Dashboard**：Next.js 监控面板（任务列表、统计概览、摘要管理、鉴别管理、任务详情追溯）
- **Docker 部署**：docker-compose 一键启动后端 + Dashboard
- **73 个测试**：覆盖全部 Agent 逻辑、工作流路由、缓存操作、RAG 基础设施

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                            │
│  ┌──────────┐    ┌─────────────────────────────────────────┐     │
│  │ REST API │    │          LangGraph Workflow               │     │
│  │ /api/*   │    │                                           │     │
│  └────┬─────┘    │  Monitor ──► Supervisor                   │     │
│       │          │                 │                          │     │
│       │          │     ┌───────────┼──────────┬─────────┐    │     │
│       │          │     ▼           ▼          ▼         ▼    │     │
│       │          │  [SumCache] Analyzer  VerifyCache  [Ign]  │     │
│       │          │     │          │       Judge       ore    │     │
│       │          │     │          ▼          │               │     │
│       │          │     │      Summarizer  ┌──┴──┐           │     │
│       │          │     │      /MapReduce  ▼     ▼            │     │
│       │          │     │          │   [VCache] Verifier      │     │
│       │          │     │          │     Hit   ($web_search)  │     │
│       │          │     │          │      │      │            │     │
│       │          │     ▼          ▼      ▼      ▼            │     │
│       │          │            Reply Agent                     │     │
│       │          └─────────────────────────────────────────┘     │
│       │                                                          │
│  ┌────┴──────────────────────────────────────┐                   │
│  │             SQLite Database                │                   │
│  │  (任务记录 / 摘要缓存 / 鉴别缓存 / 运行日志) │                   │
│  └────────────────────────────────────────────┘                   │
│  ┌────────────────────────────────────────────┐                   │
│  │             ChromaDB (向量库)               │                   │
│  │  (字幕分块 + Embedding，供 RAG 检索)        │                   │
│  └────────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
          ▲                              ▲
          │ HTTP                         │ WebSocket
          ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│  Next.js Dashboard │          │   Bilibili API   │
│  (监控面板)        │          │   (B站平台)       │
└──────────────────┘          └──────────────────┘
```

### Agent 设计

| Agent | 职责 |
|-------|------|
| **Supervisor** | 解析 @消息意图（summarize / verify / ignore），查摘要缓存，路由分发 |
| **Analyzer** | 获取视频信息 + 字幕，评估可总结性，长视频检测 + 字幕索引 |
| **Summarizer** | 生成摘要（短视频直接生成 / 长视频 Map-Reduce） |
| **VerifyCacheJudge** | 查询同视频历史鉴别，LLM 语义相似度判断是否复用缓存 |
| **Verifier** | 联网搜索 + 事实核查 + 观点生成（绕过 LangChain，直接调用 Kimi $web_search） |
| **Reply** | 格式化回复（总结/鉴别两种格式），盖楼拆分，发评论 |

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.12, FastAPI, Uvicorn |
| Agent 框架 | LangChain + LangGraph |
| LLM | Moonshot AI (Kimi K2.5, 128K context) via OpenAI 兼容 API |
| B 站 API | bilibili-api-python |
| 数据库 | SQLite (SQLAlchemy + aiosqlite) |
| 向量数据库 | ChromaDB (本地持久化) |
| Embedding | BAAI/bge-base-zh-v1.5 (本地推理，无 API 费用) |
| 前端 | Next.js 15, React 19, TypeScript, Tailwind CSS, shadcn/ui, SWR |
| 部署 | Docker + docker-compose |
| 包管理 | uv |

## 项目结构

```
BiliAgent/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── architecture.md            # 系统设计文档
├── roadmap.md                 # 开发路线图
├── src/biliagent/
│   ├── main.py                # FastAPI 入口 + Monitor 启动
│   ├── config.py              # 全局配置（环境变量）
│   ├── agents/                # Agent 定义
│   │   ├── supervisor.py      # 意图分类 + 缓存查询 + 路由
│   │   ├── analyzer.py        # 视频分析 + 长视频检测
│   │   ├── summarizer.py      # 摘要生成（含 Map-Reduce）
│   │   ├── reply.py           # 评论格式化 + 盖楼发布
│   │   ├── verify_cache_judge.py  # 鉴别缓存语义匹配
│   │   └── verifier.py        # 联网搜索 + 鉴别
│   ├── graph/                 # LangGraph 工作流
│   │   ├── state.py           # AgentState TypedDict
│   │   └── workflow.py        # StateGraph 编排
│   ├── platforms/             # 平台抽象层
│   │   ├── base.py            # PlatformBase 抽象类
│   │   └── bilibili/          # B站实现
│   ├── rag/                   # RAG 基础设施
│   │   ├── chunker.py         # 字幕分块（中文优化分隔符）
│   │   ├── vectorstore.py     # ChromaDB 封装
│   │   └── indexer.py         # 幂等索引接口
│   ├── prompts/               # 提示词模板
│   ├── models/schemas.py      # Pydantic 数据模型
│   ├── storage/               # 数据存储
│   │   ├── database.py        # SQLAlchemy 模型
│   │   ├── cache.py           # 摘要缓存
│   │   └── verify_cache.py    # 鉴别缓存
│   └── api/routes.py          # REST API 路由
├── dashboard/                 # Next.js 前端
└── tests/                     # 73 个测试
    ├── test_agents/
    ├── test_platforms/
    ├── test_graph/
    └── test_rag/
```

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

编辑 `.env` 文件，填入以下必要信息：

- **B 站 Cookie**：`BILI_SESSDATA`、`BILI_BILI_JCT`、`BILI_BUVID3`、`BILI_ACCOUNT_UID`
  - 登录 B 站后从浏览器 DevTools → Application → Cookies 获取
- **LLM API Key**：`LLM_API_KEY`
  - 从 [platform.moonshot.cn](https://platform.moonshot.cn) 获取 Moonshot API Key

### 3. 启动服务

**方式一：本地启动**

```bash
# 启动后端
uv run uvicorn biliagent.main:app --reload

# 启动前端（另一个终端）
cd dashboard && npm run dev
```

**方式二：Docker 启动**

```bash
docker-compose up -d
```

启动后：
- 后端 API: http://127.0.0.1:8000
- 健康检查: http://127.0.0.1:8000/health
- API 文档: http://127.0.0.1:8000/docs
- Dashboard: http://localhost:3000

> 首次启动会自动下载 Embedding 模型 `BAAI/bge-base-zh-v1.5`（约 400MB），请确保网络畅通。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（含 Monitor 状态、Cookie 状态） |
| GET | `/api/tasks` | 任务列表（支持分页、状态筛选） |
| GET | `/api/tasks/{id}` | 任务详情（含 Agent Trace、评论、摘要） |
| GET | `/api/stats` | 统计概览 |
| GET | `/api/summaries` | 摘要缓存列表 |
| DELETE | `/api/summaries/{id}` | 删除摘要缓存 |
| GET | `/api/verifications` | 鉴别缓存列表 |
| DELETE | `/api/verifications/{id}` | 删除鉴别缓存 |
| POST | `/api/test/trigger` | 手动触发工作流（开发测试用） |

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
| `LLM_API_KEY` | Moonshot API Key | - |
| `LLM_BASE_URL` | LLM API 地址 | `https://api.moonshot.ai/v1` |
| `LLM_MODEL` | LLM 模型名 | `kimi-k2.5` |
| `MONITOR_INTERVAL` | @消息轮询间隔（秒） | `60` |
| `SUMMARY_MAX_LENGTH` | 摘要最大字数 | `500` |
| `COMMENT_SEND_INTERVAL` | 评论发送间隔（秒） | `30` |
| `SUBTITLE_MAX_LENGTH` | 字幕截断长度 | `15000` |
| `FOLLOWER_CHECK_ENABLED` | 是否开启关注检查 | `true` |
| `VERIFY_MAX_LENGTH` | 鉴别评论最大字数 | `500` |
| `CHROMA_PERSIST_DIR` | ChromaDB 持久化目录 | `./data/chroma` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-base-zh-v1.5` |
| `CHUNK_SIZE` | 字幕分块大小（字符） | `1000` |
| `CHUNK_OVERLAP` | 分块重叠（字符） | `200` |
| `LONG_VIDEO_THRESHOLD` | 触发 Map-Reduce 的字幕长度 | `15000` |
| `QA_TOP_K` | RAG 检索返回片段数 | `5` |
| `QA_MAX_LENGTH` | 问答回答最大字数 | `500` |
| `DATABASE_URL` | 数据库连接 | `sqlite+aiosqlite:///./biliagent.db` |

## License

MIT
