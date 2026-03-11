# BiliAgent 项目路线图

> 基于 multi-agent 的B站视频自动总结机器人
> 最后更新：2026-03-10

---

## 一、项目概览

### 1.1 项目目标
开发一个多Agent协同应用：当其他用户在B站视频下@指定账号时，自动以评论方式总结该视频内容。

### 1.2 核心功能（MVP）
- 监控B站@消息通知
- 自动提取视频标题、简介、字幕
- 基于字幕内容生成视频摘要
- 以评论方式回复（超长内容自动盖楼）
- 已总结过的视频直接返回缓存结果
- 无字幕视频回复无法总结的原因

### 1.3 设计原则
- **平台可扩展**：抽象平台接口，未来可接入小红书、抖音、YouTube
- **功能可扩展**：Agent通过LangGraph编排，新功能 = 新Agent + 新路由
- **LLM可切换**：每个Agent的LLM可独立配置，通过环境变量切换

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                        │
│  ┌──────────┐    ┌──────────────────────────────────┐   │
│  │ REST API │    │      LangGraph Workflow            │   │
│  │ /api/*   │    │                                    │   │
│  └────┬─────┘    │  Monitor ──► Supervisor            │   │
│       │          │                 │                   │   │
│       │          │         ┌───────┼────────┐         │   │
│       │          │         ▼       ▼        ▼         │   │
│       │          │     [Cache]  Analyzer  [Error]     │   │
│       │          │         │       │                   │   │
│       │          │         │       ▼                   │   │
│       │          │         │   Summarizer              │   │
│       │          │         │       │                   │   │
│       │          │         ▼       ▼                   │   │
│       │          │       Reply Agent                   │   │
│       │          └──────────────────────────────────┘   │
│       │                                                  │
│  ┌────┴─────────────────────────────────┐               │
│  │           SQLite Database             │               │
│  │  (任务记录 / 摘要缓存 / 运行日志)      │               │
│  └───────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
          ▲                              ▲
          │ HTTP                         │ WebSocket
          ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│  Next.js Dashboard │          │   Bilibili API   │
│  (监控面板)        │          │   (B站平台)       │
└──────────────────┘          └──────────────────┘
```

### 2.2 Agent 设计

| 组件 | 类型 | 职责 | LLM决策点 |
|------|------|------|-----------|
| **Monitor Service** | 后台定时服务 | 轮询B站@消息通知 | 无（固定逻辑，非Agent） |
| **Supervisor Agent** | 调度Agent | 解析请求意图、查缓存、路由分发 | 判断@消息是否为有效总结请求 |
| **Analyzer Agent** | 分析Agent | 获取视频信息、提取字幕、评估可总结性 | 评估字幕质量和内容完整性 |
| **Summarizer Agent** | 生成Agent | 基于字幕生成结构化摘要 | 核心摘要生成 |
| **Reply Agent** | 回复Agent | 格式化回复、发布评论、盖楼拆分 | 智能排版和盖楼决策 |

### 2.3 LangGraph 工作流

```
                    ┌──────────────┐
                    │   START      │
                    │ (收到@通知)   │
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  Supervisor  │
                    │  查缓存/判断  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ 缓存命中  │ │ Analyzer │ │ 无效请求  │
        │ 取已有摘要 │ │ 分析视频  │ │ 忽略     │
        └─────┬────┘ └─────┬────┘ └──────────┘
              │            │
              │     ┌──────┴──────┐
              │     ▼             ▼
              │ ┌──────────┐ ┌──────────┐
              │ │有字幕     │ │无字幕     │
              │ │Summarizer│ │生成原因   │
              │ └─────┬────┘ └─────┬────┘
              │       │            │
              ▼       ▼            ▼
           ┌─────────────────────────┐
           │      Reply Agent        │
           │  格式化 → 发评论/盖楼    │
           └─────────────────────────┘
                       │
                       ▼
                    ┌──────┐
                    │ END  │
                    └──────┘
```

### 2.4 平台抽象层设计

```python
# 抽象基类 — 所有平台都实现这个接口
class PlatformBase(ABC):
    async def get_mentions()        # 获取@消息
    async def get_video_info()      # 获取视频信息
    async def get_subtitles()       # 获取字幕
    async def post_comment()        # 发布评论
    async def reply_comment()       # 回复评论

# B站实现
class BilibiliPlatform(PlatformBase): ...

# 未来扩展
class XiaohongshuPlatform(PlatformBase): ...
class DouyinPlatform(PlatformBase): ...
```

---

## 三、技术栈

### 3.1 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.12 | 主语言 |
| uv | latest | 包管理与虚拟环境 |
| LangChain | 1.0.1 | LLM调用框架 |
| LangGraph | 1.0.1 | Multi-Agent工作流编排 |
| langchain-openai | latest | Kimi 2.5 通过OpenAI兼容API接入 |
| FastAPI | latest | 后端API框架 |
| Uvicorn | latest | ASGI服务器 |
| bilibili-api-python | latest | B站API封装 |
| SQLAlchemy | latest | ORM（配合aiosqlite异步驱动） |
| aiosqlite | latest | SQLite异步驱动 |
| Pydantic | v2 latest | 数据模型验证 |
| pydantic-settings | latest | 环境变量配置管理 |

### 3.2 前端（Dashboard）

| 技术 | 版本 | 用途 |
|------|------|------|
| Next.js | 15 (App Router) | 前端框架 |
| React | 19 | UI库 |
| TypeScript | 5.x | 类型安全 |
| Tailwind CSS | 4 | 样式 |
| shadcn/ui | latest | UI组件库 |
| SWR | latest | 数据请求 |

### 3.3 LLM 配置

| 配置项 | 值 |
|--------|-----|
| LLM Provider | Moonshot AI (Kimi 2.5) |
| API Base URL | `https://api.moonshot.cn/v1` |
| 默认模型 | `moonshot-v1-32k` |
| 上下文窗口 | 32K tokens |
| 接入方式 | langchain-openai 的 ChatOpenAI，配置 base_url |

**切换LLM示例：** 每个Agent在配置中指定自己的LLM，可独立切换：
```python
# config.py 示例
AGENT_LLM_CONFIG = {
    "supervisor": {"model": "moonshot-v1-32k", "base_url": "https://api.moonshot.cn/v1"},
    "analyzer":   {"model": "moonshot-v1-32k", "base_url": "https://api.moonshot.cn/v1"},
    "summarizer": {"model": "moonshot-v1-32k", "base_url": "https://api.moonshot.cn/v1"},
    "reply":      {"model": "moonshot-v1-32k", "base_url": "https://api.moonshot.cn/v1"},
}
# 未来想给某个Agent换成GPT-4o，只需改对应配置
```

---

## 四、项目目录结构

```
BiliAgent/
├── pyproject.toml                # uv项目配置、依赖声明
├── uv.lock                       # 依赖锁定文件
├── .env.example                  # 环境变量模板
├── .env                          # 实际环境变量（不进git）
├── .gitignore
├── roadmap.md                    # 本文件
├── src/
│   └── biliagent/
│       ├── __init__.py
│       ├── main.py               # FastAPI应用入口 + 启动Monitor
│       ├── config.py             # 全局配置（读取.env）
│       │
│       ├── agents/               # Agent定义
│       │   ├── __init__.py
│       │   ├── supervisor.py     # Supervisor Agent
│       │   ├── analyzer.py       # Analyzer Agent
│       │   ├── summarizer.py     # Summarizer Agent
│       │   └── reply.py          # Reply Agent
│       │
│       ├── graph/                # LangGraph工作流
│       │   ├── __init__.py
│       │   ├── state.py          # 工作流状态定义（TypedDict）
│       │   └── workflow.py       # StateGraph编排
│       │
│       ├── platforms/            # 平台抽象层
│       │   ├── __init__.py
│       │   ├── base.py           # PlatformBase抽象类
│       │   └── bilibili/
│       │       ├── __init__.py
│       │       ├── client.py     # Bilibili API封装
│       │       └── monitor.py    # @消息轮询服务
│       │
│       ├── prompts/              # 提示词模板
│       │   ├── supervisor.txt
│       │   ├── analyzer.txt
│       │   ├── summarizer.txt
│       │   └── reply.txt
│       │
│       ├── models/               # 数据模型
│       │   ├── __init__.py
│       │   └── schemas.py        # Pydantic模型
│       │
│       ├── storage/              # 数据存储
│       │   ├── __init__.py
│       │   ├── database.py       # SQLAlchemy模型 + 初始化
│       │   └── cache.py          # 摘要缓存操作
│       │
│       └── api/                  # FastAPI路由
│           ├── __init__.py
│           └── routes.py         # REST API端点
│
├── dashboard/                    # Next.js前端（Phase 2）
│   ├── package.json
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx              # Dashboard首页
│   │   └── api/                  # Next.js API Routes（代理转发）
│   ├── components/
│   │   ├── TaskList.tsx          # 任务列表
│   │   ├── StatsCard.tsx         # 统计卡片
│   │   └── AgentStatus.tsx       # Agent状态
│   └── ...
│
└── tests/                        # 测试
    ├── __init__.py
    ├── test_agents/
    ├── test_platforms/
    └── test_graph/
```

---

## 五、数据库设计（SQLite）

### 5.1 表结构

```sql
-- 任务记录表：记录每一次@触发的任务
CREATE TABLE tasks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    platform        TEXT NOT NULL DEFAULT 'bilibili',   -- 平台
    mention_id      TEXT NOT NULL UNIQUE,                -- @消息唯一ID（去重）
    video_id        TEXT NOT NULL,                       -- 视频ID（如BV号）
    user_id         TEXT NOT NULL,                       -- @我的用户ID
    user_name       TEXT,                                -- @我的用户昵称
    status          TEXT NOT NULL DEFAULT 'pending',     -- pending/processing/completed/failed
    error_message   TEXT,                                -- 失败原因
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 摘要缓存表：同一视频只总结一次
CREATE TABLE summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    platform        TEXT NOT NULL DEFAULT 'bilibili',
    video_id        TEXT NOT NULL,                       -- 视频ID
    video_title     TEXT,                                -- 视频标题
    summary_text    TEXT NOT NULL,                       -- 生成的摘要
    has_subtitles   BOOLEAN NOT NULL DEFAULT 1,          -- 是否有字幕
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(platform, video_id)                           -- 同平台同视频唯一
);

-- 评论记录表：记录已发布的评论
CREATE TABLE comments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         INTEGER NOT NULL REFERENCES tasks(id),
    platform        TEXT NOT NULL DEFAULT 'bilibili',
    comment_id      TEXT,                                -- 平台返回的评论ID
    content         TEXT NOT NULL,                       -- 评论内容
    floor_number    INTEGER NOT NULL DEFAULT 1,          -- 盖楼楼层号
    posted_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Agent执行追溯表：记录每个Agent节点的执行详情（供Dashboard任务详情页使用）
CREATE TABLE agent_traces (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         INTEGER NOT NULL REFERENCES tasks(id),
    agent_name      TEXT NOT NULL,                       -- supervisor/analyzer/summarizer/reply
    input_data      TEXT,                                -- 该节点的输入（JSON）
    output_data     TEXT,                                -- 该节点的输出（JSON）
    duration_ms     INTEGER,                             -- 执行耗时（毫秒）
    status          TEXT NOT NULL DEFAULT 'success',     -- success/failed/skipped
    error_message   TEXT,                                -- 错误信息
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 六、开发阶段

### Phase 1：项目基础搭建（第1-2天）

**目标：** 项目骨架跑通，能启动FastAPI服务

- [x] 1.1 用 `uv init` 初始化项目
- [x] 1.2 配置 `pyproject.toml`，添加所有依赖
- [x] 1.3 创建目录结构（按第四节）
- [x] 1.4 编写 `.env.example` 和 `.gitignore`
- [x] 1.5 编写 `config.py`（读取环境变量：B站Cookie、LLM API Key等）
- [x] 1.6 编写 `main.py`（FastAPI骨架 + 健康检查接口）
- [x] 1.7 编写 `storage/database.py`（SQLAlchemy模型 + 建表）
- [x] 1.8 验证：`uvicorn biliagent.main:app` 能启动，`/health` 返回OK
- [ ] 1.9 初始化 git 仓库，首次提交

**实际安装命令：**
```bash
uv init --name biliagent --python 3.12
uv sync  # 71个包，含 langchain 1.2.10, langgraph 1.0.10
```

**变更记录：**
- APScheduler 4.0 未发布稳定版，Monitor 轮询改用 Python 原生 `asyncio` 循环（零额外依赖）
- 新增 `pydantic-settings` 依赖（环境变量管理）

### Phase 2：平台层 + B站API对接（第3-4天）

**目标：** 能成功获取B站@消息和视频字幕

- [x] 2.1 编写 `platforms/base.py`（PlatformBase抽象类）
- [x] 2.2 编写 `platforms/bilibili/client.py`（封装bilibili-api-python）
  - `get_mentions()` — 获取@我的消息列表
  - `get_video_info(bvid)` — 获取视频标题、简介
  - `get_subtitles(bvid)` — 获取视频字幕
  - `post_comment(bvid, text)` — 发布评论
  - `reply_comment(bvid, root_id, text)` — 回复评论（盖楼用）
- [x] 2.3 编写 `platforms/bilibili/monitor.py`（asyncio轮询@消息服务）
- [x] 2.4 编写 `models/schemas.py`（Pydantic数据模型）
- [ ] 2.5 手动测试：获取@消息列表、获取指定视频字幕、发一条测试评论（需填入真实Cookie）
- [x] 2.6 编写 `storage/cache.py`（摘要缓存的增删查）

**注意事项：**
- bilibili-api-python 需要用 Credential 对象传入 Cookie（sessdata, bili_jct, buvid3）
- @消息通过 B站"我的消息"API获取，需要用 `bilibili_api.session` 模块
- 发评论需要 bili_jct（CSRF token），确保 Cookie 中包含

### Phase 3：Agent开发（第5-7天）

**目标：** 4个Agent各自能独立工作

- [x] 3.1 编写提示词模板 `prompts/*.txt`（4个模板）
- [x] 3.2 编写 `agents/__init__.py`（LLM工厂 + 提示词加载）
- [x] 3.3 编写 `agents/supervisor.py`（意图判断 + 缓存查询 + 路由）
- [x] 3.4 编写 `agents/analyzer.py`（视频信息 + 字幕 + 可总结性评估）
- [x] 3.5 编写 `agents/summarizer.py`（摘要生成 + 硬截断兜底）
- [x] 3.6 编写 `agents/reply.py`（格式化 + 盖楼拆分 + 发评论）
- [x] 3.7 验证全部 Agent 导入和 LLM 初始化正常

**Supervisor 提示词核心逻辑：**
```
你是BiliAgent的调度员。用户在B站视频下@了我们的账号。
请分析这条@消息，判断用户是否在请求视频总结。
如果是，返回 {"action": "summarize", "video_id": "xxx"}
如果不确定或不是，返回 {"action": "ignore", "reason": "xxx"}
```

**Summarizer 提示词核心逻辑：**
```
你是一个视频内容总结专家。请基于以下信息生成简洁的视频摘要：
- 标题：{title}
- 简介：{description}
- 字幕内容：{subtitles}

要求：
1. 摘要严格控制在500字以内（适配B站评论区）
2. 用清晰的结构（要点列表）
3. 语言风格适合B站评论区（简洁、友好）
4. 开头用一句话概括视频核心内容
```

### Phase 4：LangGraph工作流编排（第8-9天）

**目标：** 把4个Agent串成完整的工作流

- [x] 4.1 编写 `graph/state.py`（AgentState TypedDict，含 traces 追溯字段）
- [x] 4.2 编写 `graph/workflow.py`（StateGraph 4节点 + 2条件路由，含 trace 记录）
- [x] 4.3 集成 Monitor → handle_mention → workflow.ainvoke
- [x] 4.4 验证图编译：6节点（__start__, supervisor, analyzer, summarizer, reply, __end__）

### Phase 5：MVP联调 + 端到端测试（第10-11天）

**目标：** 完整跑通"检测@→总结→回复"流程

- [x] 5.1 启动FastAPI服务 + Monitor轮询 + /health 返回 monitor_running
- [ ] 5.2 用小号在B站@主账号，验证完整流程（需填入真实Cookie和LLM API Key）
- [ ] 5.3 测试场景：
  - 正常视频（有字幕）→ 应生成摘要并评论
  - 无字幕视频 → 应回复无法总结的原因
  - 重复@同一视频 → 应命中缓存，直接回复
  - 非总结请求的@ → 应忽略
  - 超长摘要 → 应自动盖楼
- [ ] 5.4 修复bug，调整提示词
- [x] 5.5 全链路英文日志（每个 Agent 节点 + Monitor + Workflow）
- [x] 5.6 API 路由完整实现：
  - `GET /api/tasks` — 任务列表（支持分页、状态筛选）
  - `GET /api/tasks/{id}` — 任务详情（含 traces + comments + summary 完整追溯）
  - `GET /api/stats` — 统计概览
  - `GET /api/summaries` — 摘要缓存列表
  - `DELETE /api/summaries/{id}` — 删除摘要缓存
  - `POST /api/test/trigger` — 手动触发工作流（开发测试用，不依赖真实@消息）

### Phase 6：Dashboard开发（第12-15天）

**目标：** 可视化监控面板

- [x] 6.1 在 `dashboard/` 下用 `npx create-next-app@latest` 初始化 Next.js 项目
- [x] 6.2 安装 shadcn/ui + Tailwind CSS
- [x] 6.3 开发Dashboard页面：
  - **首页概览**：统计卡片（总任务数、成功率、今日处理数、Cookie状态）
  - **任务列表**：状态、视频标题、时间，可点击进入详情
  - **任务详情页（重点）**：追溯每个任务的完整生成过程
    - Agent Trace 时间线（每个节点的 Input/Output JSON、耗时、状态）
    - 生成的摘要原文
    - Reply Agent最终发布的评论内容
    - 错误信息（如有）
  - **摘要缓存管理**：查看/删除已缓存的摘要
- [x] 6.4 后端API完整实现（任务详情含 traces/comments/summary 完整追溯）
- [x] 6.5 对接后端API（使用SWR做数据轮询刷新）
- [x] 6.6 联调测试

### Phase 7：优化与加固（第16-18天）

**目标：** 生产级质量

- [x] 7.1 异常处理完善（网络超时、API限流、Cookie过期等）
- [x] 7.2 Cookie过期检测与告警
- [x] 7.3 评论发送频率控制（每条评论间隔≥30秒）
- [x] 7.4 字幕文本截断策略（超长字幕只取前15000字符送LLM）
- [x] 7.5 添加单元测试（核心Agent逻辑）
- [x] 7.6 添加集成测试（Mock B站API）
- [x] 7.7 编写 README.md

**实现记录：**
- 7.1: `invoke_llm_with_retry()` — 最多3次重试，指数退避(2s/4s/8s)，识别 timeout/429/5xx；LLM 请求超时 60s
- 7.2: `check_credential()` 启动时检测 + 运行中自动感知；Dashboard Cookie Status 卡片 + 红色告警横幅
- 7.5+7.6: 49 个测试全部通过（24 单元 + 25 集成）
- 7.7: 完整 README.md

### Phase 8：关注检查（权限拦截）

**目标：** 仅为已关注本账号的用户提供视频总结服务，未关注用户收到引导关注提示

**方案：** 前置拦截 — 在 `handle_mention()` 入口处检查关注关系，未关注者不进入 LangGraph 工作流，直接回复提示文案

**处理流程：**
```
收到@消息 → 检查用户是否关注本账号
  ├─ 未关注 → 直接回复"请关注后使用" → 任务标记 not_follower → END
  └─ 已关注 → 进入正常 LangGraph 工作流（Supervisor → ...）
```

**回复文案（傲娇人设）：**
- 已关注（总结回复）：`@用户名 哼，既然你都诚心诚意地召唤我了，那我就大发慈悲地给你总结一下吧！\n\n[总结内容]\n\n（拿走不谢！下次还要找我哦~ [傲娇]）`
- 未关注（求关注）：`@用户名 喂喂！连个关注都不点就想使唤我？[生气] 抓到一只企图白嫖野生总结的B友！快乖乖点上关注，不然本课代表要罢工啦！(〃＞目＜)`

- [ ] 8.1 `platforms/base.py` — 抽象基类新增 `check_is_follower(user_id: str) -> bool` 方法
- [ ] 8.2 `platforms/bilibili/client.py` — 实现关注关系检查（调用 B站用户关系 API）
- [ ] 8.3 `config.py` — 新增 `FOLLOWER_CHECK_ENABLED` 开关（默认开启）和 `NOT_FOLLOWER_REPLY` 回复文案配置
- [ ] 8.4 `main.py` — `handle_mention()` 中 `workflow.ainvoke()` 之前插入关注检查逻辑
- [ ] 8.5 `storage/database.py` — Task status 新增 `not_follower` 状态值
- [ ] 8.6 `api/routes.py` — Dashboard 统计接口适配 `not_follower` 状态
- [ ] 8.7 补充单元测试 + 集成测试
- [ ] 8.8 端到端验证：用未关注小号@机器人，确认收到关注提示

**设计要点：**
- 不修改 LangGraph 图结构，关注检查为硬逻辑（非 LLM 判断）
- 未关注用户不消耗 LLM API 配额
- `FOLLOWER_CHECK_ENABLED=false` 可关闭此功能（调试用）

---

## 七、环境变量配置

```env
# .env.example

# ===== Bilibili =====
BILI_SESSDATA=your_sessdata_here
BILI_BILI_JCT=your_bili_jct_here
BILI_BUVID3=your_buvid3_here
BILI_ACCOUNT_UID=your_uid_here

# ===== LLM (Kimi 2.5 / Moonshot) =====
LLM_API_KEY=your_moonshot_api_key_here
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL=moonshot-v1-32k

# ===== App =====
MONITOR_INTERVAL=60
SUMMARY_MAX_LENGTH=500
COMMENT_SEND_INTERVAL=30
SUBTITLE_MAX_LENGTH=15000
FOLLOWER_CHECK_ENABLED=true
NOT_FOLLOWER_REPLY=喂喂！连个关注都不点就想使唤我？[生气] 抓到一只企图白嫖野生总结的B友！快乖乖点上关注，不然本课代表要罢工啦！(〃＞目＜)

# ===== Database =====
DATABASE_URL=sqlite+aiosqlite:///./biliagent.db
```

---

## 八、评论输出策略

### 8.1 摘要长度控制
- LLM生成摘要时严格限制在 **500字以内**
- 通过提示词硬约束 + 后处理截断双重保障
- 500字以内的评论通常不会触发B站字数限制，因此大多数情况下单条评论即可

### 8.2 盖楼兜底策略（安全网）
极端情况下（如LLM输出超预期），仍保留盖楼能力：

1. **首条评论**（主楼）：摘要开头部分 + `「续 ↓」`
2. **后续评论**（回复主楼）：`「第2部分」` + 后续内容
3. **末条评论**：内容 + `「完」`
4. **拆分规则**：按段落优先拆分，避免把一句话拆成两半

---

## 九、已确认事项

| 事项 | 决定 |
|------|------|
| LLM模型 | `moonshot-v1-32k`（32K上下文窗口） |
| 摘要输出长度 | 严格限制500字以内 |
| 轮询频率 | 60秒/次 |
| 部署环境 | 本地运行（开发者电脑） |
| Dashboard | 需支持任务详情追溯（查看每个任务的完整生成过程） |

---

## 十、风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| B站Cookie过期 | 所有功能瘫痪 | 定期检测 + Dashboard告警 |
| 评论被B站限流/封禁 | 无法发评论 | 控制频率、内容差异化 |
| LLM API调用失败 | 无法生成摘要 | 重试机制（最多3次） + 错误日志 |
| 字幕过长超出LLM上下文 | 截断导致摘要不完整 | 智能截断 + 在摘要中注明 |
| B站API变更 | 接口不可用 | bilibili-api-python社区维护，关注更新 |
| 关注关系API限流 | 频繁查询被限制 | 可考虑短期缓存关注状态（如缓存5分钟） |
