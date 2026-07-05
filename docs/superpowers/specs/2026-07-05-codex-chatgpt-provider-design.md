# Codex ChatGPT Provider Design

**状态：** 已确认的设计，尚未实施

**日期：** 2026-07-05

**仓库：** `lgravity327/TradingAgents`

**目标 provider key：** `codex_chatgpt`

## 1. 目标

为 TradingAgents 增加一个显式选择的 `codex_chatgpt` LLM provider。该 provider 通过本机官方 Codex CLI 已存在的 “Sign in with ChatGPT” 身份执行模型请求，不要求任何 LLM API key，并尽可能保持当前 LangGraph 多智能体流程、工具调用和结构化输出行为不变。

这里的“使用 ChatGPT 额度”严格定义为：使用 ChatGPT 账号所包含的 **Codex 使用额度**。它不等同于把 ChatGPT 网页消息额度转换为 OpenAI API 额度，也不意味着 ChatGPT 网页私有接口成为可编程 API。

## 2. 已核验前提

- 本地仓库位于 `/Users/gang.luo/Antigravity/TradingAgentsGH`，本地 HEAD 与 `origin/HEAD` 均为 `85946c2f60768ab2dae23a5a36cd927662feef94`。
- 当前机器安装 `codex-cli 0.142.5`。
- `codex login status` 当前返回 `Logged in using ChatGPT`。
- 当前 CLI 提供非交互命令 `codex exec`，并支持 `--ephemeral`、`--sandbox read-only`、`--output-schema`、`--json`、`--output-last-message`、`--ignore-user-config` 和 `--skip-git-repo-check`。
- TradingAgents 的 LLM 抽象不仅使用普通 `invoke`，还依赖 `bind_tools` 和 `with_structured_output`。仅删除 API key 检查不能满足现有执行语义。
- OpenAI 官方将 ChatGPT 与 API 作为独立计费产品；本设计不尝试用 ChatGPT 登录凭据伪装 OpenAI API 请求。

## 3. 非目标与禁止项

- 不读取、复制、解析或提交 `~/.codex/auth.json` 及任何 OAuth token。
- 不读取浏览器 Cookie，不调用 ChatGPT 私有 `backend-api`，不部署模拟 ChatGPT 网页会话的反向代理。
- 不修改现有 `openai`、`anthropic`、`google`、OpenAI-compatible 或本地 provider 的认证行为。
- 不把 `codex_chatgpt` 设为默认 provider；现有用户行为必须保持不变。
- 不保证 Codex CLI 与 OpenAI API 模型具有相同系统提示、模型列表、token 统计、延迟或输出质量。
- 不在登录失效或额度耗尽时静默回退到付费 API provider。

## 4. 方案比较

### 4.1 采用：仓库内 Codex CLI provider

在 TradingAgents 的 provider 工厂中注册 `codex_chatgpt`，实现一个 LangChain-compatible chat model。模型调用由受控 subprocess 执行官方 `codex exec`，并将 Codex 的结构化结果转换为 `AIMessage`、tool calls 或 Pydantic 对象。

选择原因：它能复用已登录的官方 Codex 身份，认证边界清晰；同时改动集中在 LLM provider 层，不需要重写 LangGraph 工作流。

### 4.2 未采用：本地 OpenAI-compatible gateway

可以另起本地 HTTP 服务，将 `/v1/chat/completions` 转换为 `codex exec`。这会让 TradingAgents 复用现有 `openai_compatible` provider，但 gateway 仍需重新实现 tool calls、structured output、进程管理和错误映射，还增加服务生命周期、端口和部署复杂度。除非后续需要供多个项目共享，否则当前范围不值得引入。

### 4.3 拒绝：ChatGPT Cookie 或私有接口

该路径不是受支持的开发接口，认证格式可能随时变化，还会扩大凭据泄漏和账号限制风险。设计明确禁止实现。

### 4.4 可选回退但不属于本项目：Ollama

如果核心目标只是“无 API key、无远端按 token 计费”，仓库已经支持 Ollama。它最稳定，但不使用 ChatGPT/Codex 额度，因此不是本次目标实现。

## 5. 架构

新增实现分为三个边界明确的组件：

1. **Codex CLI runner**：只负责构造安全命令、执行 subprocess、管理临时目录、解析 JSONL/最终输出、处理超时与退出码。
2. **Codex LangChain chat model**：负责消息序列化、普通回答、tool-call 协议、structured output 解析，以及转换为 LangChain 类型。
3. **TradingAgents provider integration**：负责 factory 注册、CLI provider 选择、模型选项、API-key 跳过和文档。

现有图编排、agent prompt、ToolNode 和报告渲染不感知 subprocess 细节。

## 6. Codex CLI 执行边界

每次 LLM invocation 启动一个新的、无会话持久化的 Codex 进程。基准命令语义如下：

```text
codex exec
  --ephemeral
  --ignore-user-config
  --ignore-rules
  --sandbox read-only
  --skip-git-repo-check
  --cd <empty-temporary-directory>
  --json
  --output-schema <temporary-schema-file>
  --output-last-message <temporary-output-file>
  [--model <explicit-model>]
  -
```

完整 prompt 从 stdin 传入，避免命令行长度限制和 shell quoting 风险。进程以参数数组启动，禁止 `shell=True`。`--ignore-user-config` 用于隔离个人模型、MCP 和行为配置；Codex CLI 的帮助信息明确说明认证仍使用 `CODEX_HOME`。工作目录指向新建的空临时目录，避免 Codex 自动读取 TradingAgents 仓库内容或仓库级指令。

runner 必须监控 `--json` 事件。若 Codex 在一次模型适配请求中尝试执行 shell、文件或网络工具，该 invocation 视为协议失败；provider 不允许 Codex 自行完成 TradingAgents 工具的职责。

临时 schema、最终输出和空工作目录在 invocation 结束后删除。日志不得记录完整 prompt、OAuth 信息或环境变量值。

## 7. 调用协议

### 7.1 普通 `invoke`

LangChain messages 被序列化为带明确 role 边界的文本。输出 schema 固定为：

```json
{
  "type": "object",
  "properties": {
    "content": {"type": "string"}
  },
  "required": ["content"],
  "additionalProperties": false
}
```

provider 将 `content` 转成 `AIMessage(content=...)`。

### 7.2 `bind_tools`

LangChain tool 名称、说明和 JSON Schema 参数被加入 prompt。Codex 只能返回以下互斥结果之一：

```json
{
  "mode": "final",
  "content": "最终回答",
  "tool_calls": []
}
```

或：

```json
{
  "mode": "tool_calls",
  "content": "",
  "tool_calls": [
    {
      "id": "call_<stable-random-id>",
      "name": "get_stock_data",
      "args": {"symbol": "AAPL"}
    }
  ]
}
```

provider 校验 tool 名称必须存在于本次绑定集合，`args` 必须满足对应 schema，然后构造 LangChain `AIMessage.tool_calls`。现有 LangGraph `ToolNode` 执行真实工具，工具结果通过下一轮 messages 再交给 provider。Codex CLI 自身不执行这些工具。

未知工具、无效参数、同时返回 final 和 tool calls、空 tool-call 列表均作为协议错误，不猜测修复。

### 7.3 `with_structured_output`

Pydantic schema 转换为 JSON Schema 后直接传给 `codex exec --output-schema`。最终 JSON 通过目标 Pydantic model 校验并返回。校验失败抛出异常，由仓库现有 `invoke_structured_or_freetext` 逻辑执行一次 free-text fallback。

当前 TradingAgents 不在同一个调用上同时组合 `bind_tools` 与 `with_structured_output`；本版本不扩展该未使用组合。

## 8. 模型选择

`codex_chatgpt` 的 quick/deep 模型菜单都至少提供：

- `Codex account default`，内部值为 `default`，runner 不传 `--model`。
- `Custom model ID`，由用户显式输入并作为 `--model` 参数传入。

不在仓库中硬编码可能快速变化的 Codex 模型列表。默认 provider 仍为 `openai`；选择 `codex_chatgpt` 时 quick/deep 默认均可使用 `default`。

## 9. 认证、额度与错误处理

创建 `codex_chatgpt` client 时先运行低成本的本地 `codex login status`，只判断退出码和稳定状态文本，不访问凭据文件。

错误必须映射为用户可操作的信息：

- 找不到可执行文件：提示安装或升级官方 Codex CLI。
- 未登录：提示执行 `codex login` 并选择 ChatGPT 登录。
- 额度或 rate limit：明确说明这是 Codex 额度限制，不回退到 API。
- 不支持的 model：显示请求的 model ID，并建议改用 `default`。
- 超时：终止整个子进程组，报告配置的 timeout。
- 非零退出码：保留经过脱敏、长度受限的 stderr 摘要。
- 输出文件缺失、JSON 无效或 schema 不匹配：报告协议错误。
- Codex 自行发起工具操作：报告隔离违规并终止当前 invocation。

默认 timeout 为每次 invocation 300 秒，并允许通过 provider 配置覆盖。smoke 必须报告实测耗时；若正常调用在本机多次接近该上限，则判为性能风险并停止完整浅层分析，而不是继续放大 timeout。

## 10. 回调、统计和可观测性

LangChain `BaseChatModel` 负责触发现有 start/end/error callbacks。若 Codex JSONL 提供可稳定解析的 token usage，则写入 `ChatResult.llm_output`；若当前 CLI 未提供，明确标记 usage unavailable，不伪造 token 数。

provider 记录以下非敏感诊断字段：provider、requested model、elapsed time、exit classification、是否 structured/tool-bound。默认不记录 prompt 和完整模型输出。

## 11. 配置与兼容性

需要注册的公开配置行为：

- `llm_provider = "codex_chatgpt"`
- `deep_think_llm = "default"` 或显式 Codex model ID
- `quick_think_llm = "default"` 或显式 Codex model ID
- `backend_url` 对该 provider 无效且不得传入 CLI
- `TRADINGAGENTS_LLM_PROVIDER=codex_chatgpt` 可用于非交互选择

`api_key_env` 为该 provider 返回 `None`，CLI 不显示 key prompt。该 `None` 只表示 Codex CLI 自行管理认证，不代表 provider 匿名访问。

## 12. Go/No-Go 验证

正式接入前先完成独立 smoke 验证。只有以下条件全部满足才继续修改 factory、CLI 和文档：

1. 未设置任何 LLM API key 时，普通回答成功返回非空 `AIMessage.content`。
2. Pydantic schema 调用返回可验证对象。
3. 绑定一个无副作用的测试 tool 后，模型能返回合法 tool call；执行伪 tool 并回传 ToolMessage 后，模型能返回 final answer。
4. JSONL 中没有 Codex 自身 shell、文件或网络工具执行事件。
5. 登录失效、timeout、无效 model 和 malformed output 均产生确定、脱敏的错误。
6. smoke 记录每次 invocation 的耗时和可用 usage 数据，用于决定默认 timeout，并在继续实施前报告预估的完整浅层分析成本。

任一功能性或隔离条件失败即为 No-Go。No-Go 后不实现 Cookie/private API 备选；用户只能选择修复官方 Codex CLI 路径、使用现有 Ollama，或恢复正式 API provider。

## 13. 测试策略

### 13.1 单元测试

- runner 命令参数、stdin、安全选项和环境继承。
- login status 的 logged-in、logged-out、missing binary 分支。
- JSONL 和 final-output 解析。
- timeout、进程退出、stderr 脱敏和临时目录清理。
- messages role/内容序列化。
- plain、tool-call 和 Pydantic structured-output 转换。
- tool 名称与 args schema 拒绝路径。
- factory、model catalog、API-key 映射和 CLI provider 表注册。

所有单元测试 mock subprocess，不消耗 Codex 额度。

### 13.2 回归测试

运行现有 provider registry、API-key、vendor routing、structured agents、CLI config precedence、env override 与模型验证测试，证明新增 provider 未改变其他 provider。

### 13.3 Opt-in 集成测试

真实 smoke 必须通过显式环境开关启动，默认测试套件不得消耗额度。测试前验证 `codex login status`；结果报告调用次数、耗时、错误分类和是否观察到 usage。

## 14. 风险与约束

- **产品语义风险：** Codex 是 agent surface，不是通用 Chat Completions API；系统行为可能影响金融分析输出。
- **协议稳定性风险：** `codex exec` CLI 选项和 JSONL 事件格式可能随版本变化，runner 必须做版本/能力预检并 fail closed。
- **性能风险：** 每个 LLM invocation 启动独立进程，完整多智能体运行可能显著变慢。
- **额度风险：** 多轮 analyst/tool/debate 会消耗多次 Codex 调用，额度规则由 ChatGPT 计划决定。
- **工具调用风险：** tool calls 是通过结构化输出模拟，不是 OpenAI API 原生 function calling。
- **统计风险：** Codex CLI 未承诺提供与 API 相同的 token usage 结构。
- **金融可靠性风险：** 切换 provider 不提高分析正确性；现有市场数据验证和报告约束仍必须保留。

## 15. 完成标准

实现完成需要同时满足：

- 用户可以在 CLI 或 programmatic config 中选择 `codex_chatgpt`。
- 未设置 LLM API key、但官方 Codex CLI 已使用 ChatGPT 登录时，可完成一次浅层 TradingAgents 分析并生成最终 rating/report。
- analyst 工具循环以及 Research Manager、Trader、Portfolio Manager 的 structured output 均可运行。
- 登录/额度/超时/协议错误可诊断，且不会产生付费 API 静默回退。
- 现有 provider 测试保持通过。
- 仓库和日志中不存在 OAuth token、Cookie 或其他登录凭据。

## 16. 回滚策略

新增 provider 为独立 opt-in 路径。若 CLI 兼容性或输出质量不可接受，可以删除 `codex_chatgpt` 注册、实现和测试，不需要迁移现有配置或修改其他 provider。默认 `openai` 行为始终保持不变。
