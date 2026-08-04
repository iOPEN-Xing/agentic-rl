# DeepSeek V4 Flash User Simulator 全量生成与 SFT 导出审计

> 状态：数据生成与静态/数据合同审计完成；Qwen3-14B LoRA、在线 rollout 和主 Policy RL 尚未运行。本文记录 2026-08-04 的真实全量结果，不把尚未发生的训练收益写成结论。

## 1. 最终可用产物

最终自然分布包含 1,513 个 seen-task runtime-reachable 状态：

- `outputs/user_simulator_data/full/v1.2.jsonl`：DeepSeek 原始全量生成，1,513/1,513 通过最终自动门禁；
- `outputs/user_simulator_data/full/v1.2_audited.jsonl`：70 个决策分歧全部复核后的训练源；
- `outputs/user_simulator_data/full/v1.2_audited.report.json`：语义修改、决策分布、provenance 与 SHA256；
- `outputs/user_simulator_data/sft/train.jsonl`：项目训练器可直接读取的 rich train；
- `outputs/user_simulator_data/sft/eval.jsonl`：trial07 自然分布 eval；
- `outputs/user_simulator_data/sft/train_trl.jsonl`、`eval_trl.jsonl`：每行严格只有 `{"messages": [...]}` 的 TRL conversational JSONL；
- `outputs/user_simulator_data/sft/manifest.json`：源文件、对齐检查、行数与所有导出文件 SHA256。

这些数据位于 gitignored `outputs/`，不会把大体量样本或任何密钥提交到 Git。代码、合同和本报告进入版本控制。

## 2. 为什么是 1,513 / 420，而不是早期估算的 1,527 / 424

重建 case 时发现，历史轨迹中有 14 个 seen 状态和 4 个 holdout 状态发生在 tool-only turn 之后。role flip 后，它们的 `student_messages` 已经以 `assistant`（模拟 Customer）结尾；若再训练下一条 Customer target，就会教模型连续说两次。

当前 `TauBenchInteraction` 只有在 Agent 产生自然语言 `RESPOND` 时才调用 User Simulator，tool call 后不会立即调用。因此这 18 个状态在当前 runtime 不可达，必须过滤。最终合同为：

| Split | 数量 | 是否可发给 Teacher | 用途 |
|---|---:|---|---|
| curated pilot | 16 | 是 | 关键边界稳定性门禁 |
| seen train natural states | 1,513 | 是 | Teacher generation / SFT |
| benchmark holdout | 420 | 否 | unseen 审计，绝不生成或微调 |

1,513 条 seen prefix 全部以 `user` role 结束，表示“最新一条 Agent 可见文本”，随后才追加 Student 的最后一个 `assistant` target。

## 3. Teacher 配置与信息边界

全量最终配置：

```json
{
  "model": "deepseek-v4-flash",
  "thinking": false,
  "temperature": 0.3,
  "top_p": 0.9,
  "max_tokens": 1600,
  "prompt_version": "tau-airline-usim-teacher-v1.5"
}
```

v1.5 的关键变化不是再加一句“不要泄漏”，而是改变信息架构：DeepSeek 请求只包含 `scenario + observable conversation`，gold actions、tool results、trajectory reward 和 hidden identifiers 完全不进入请求。privileged 数据只留在本地 deterministic QA，用于检查“回复是否出现了可见历史里从未出现的 ID”。

这修复了 v1.4 全量试跑的根因：v1.4 虽然文字上禁止使用 privileged reference，但 Teacher 实际看到了该块，1,513 条中有 58 条泄漏 reservation/payment 等 hidden entity。仅靠 prompt 禁令无法消除信息捷径；正确方案是物理移除不该观察的信息。

v1.5 全量最终结果：

| 指标 | 结果 |
|---|---:|
| 自然 case | 1,513 |
| accepted | 1,513 / 1,513 |
| rejected / quarantined | 0 / 0 |
| privileged entity leak | 0 |
| response 长度 | min 10 / median 110 / p95 243 / max 319 chars |
| generation attempts | 1,655 |
| retried cases | 56 |
| 最大单 case attempts | 8 |
| canonical final-row prompt tokens | 3,011,234 |
| canonical final-row completion tokens | 282,631 |
| canonical final-row total tokens | 3,293,865 |
| all-attempt prompt tokens（可精确重建） | 3,362,525 |
| all-attempt total tokens（保守下界） | 3,645,156 |

前 1,513 次请求后有 56 条进入定点重采样；最终 11 条是 `continue + unresolved_goals=[] + terminal courtesy` 的内部矛盾，1 条长期超长。实现没有降低门禁：只有当结构化状态表示无未决目标、自然文本又明确说 “that's all / all I needed” 时，才把 courtesy 规范化为 runtime 唯一合法的 `###STOP###`；实质性回复若漏填 unresolved goal 仍会 quarantine。超长样本继续调用 API，直到 320 字符上限内。

usage 的口径必须单独说明：canonical JSONL 会丢弃旧 retry row，所以 3,293,865 是最终 1,513 个响应的 usage，不是完整账单。prompt 不随同一 case 的重试变化，可由最终行的 `generation_attempt` 重建所有 1,655 次请求的 3,362,525 个 prompt tokens。旧 retry completion/cache usage 已不可恢复，因此 3,645,156 只是“全部 prompt + 最终保留 completion”的总 token 下界，不伪造为精确费用。未来若需要精确账单，应保留独立 append-only attempt ledger 或使用 provider billing export。

## 4. 为什么自动门禁通过后还要语义审计

JSON schema、ID 泄漏和长度都通过，不等于 compound goal 一定判断正确。全量中有 70 个 case 的历史 next-turn decision 与新 Teacher 不一致：

- historical CONTINUE → Teacher STOP：18；
- historical STOP → Teacher CONTINUE：52。

历史标签不是 gold，新 Teacher 也不是 gold，所以不能强制二者相等。逐条读取 scenario、完整 observable history、最新 Agent 文本和两种候选后，审计结果为：

| 处理 | 数量 | 典型错误 |
|---|---:|---|
| `canonicalize_stop` | 40 | 已给出金额仍重复确认；目标完成后说 thanks；凭空要求 transfer/证书条款；缺 ID 且已约定稍后回来 |
| `restore_continue` | 4 | 漏掉 passenger-name、gift-card、错误舱位核对、剩余 reservation cancellation |
| `rewrite_continue` | 2 | 决策方向对，但回复自相矛盾或没有询问真正缺失的总数 |
| `retain_deepseek` | 24 | 复核确认新 Teacher 正确修正历史 premature STOP，原 target 不改 |
| DeepSeek 原样保留 | 1,467 | 无需修改 |

具体 case 和理由固化在 `apply_full_batch_audit.py`，每个修改后的 rich record 都写入 `target_provenance=human_semantic_audit`、audit version 与 action；没有静默改数据，也没有把修订样本宣称为纯 Teacher 输出。

最终自然决策分布：

| Decision | 数量 |
|---|---:|
| CONTINUE | 1,313 |
| STOP | 200 |

终止原因进一步拆为 181 个 `goal_satisfied`、8 个 `cannot_continue`、11 个 `goal_failed`。这很重要：`###STOP###` 是 episode terminal action，不等价于 task success；无法继续和 fallback 全部失败也需要正确结束，但不能在元数据中伪装成成功。

## 5. 两个具体 case

### 5.1 提前 STOP：task 4 的 gift-card 目标

Scenario 同时要求 cabin/baggage/passenger/payment。最新 Agent 只说 cabin、passenger、bags 已完成，随后问是否还需要帮助；DeepSeek 一度输出 STOP。但当前可见历史里用户明确偏好 gift card，Agent 尚未处理付款。最终 target 恢复为：

```text
Yes, I'd like to use my gift card for the payment.
```

若保留 STOP，User Simulator 会把“数据库部分动作完成”误学成“所有用户目标完成”，主 Policy 反而学不到 payment fallback。

### 5.2 Delayed STOP：Agent 已明确给出 Mastercard 总额

在 task 9 中，最新 Agent 已写明：

```text
Total amount charged to your Mastercard: $2,033
```

Teacher 却再次询问 Mastercard 总额。这既不增加新信息，也会给 RL 制造一个无意义状态。最终 target 规范化为精确 `###STOP###`。这里修的是 termination latency，而不是把 Agent 漏报金额的 case 强行结束；如果 Agent 没写 `$2,033`，相邻状态仍保留 CONTINUE 追问。

## 6. 与训练时 User Simulator 完全对齐

本次不是只检查“看起来像 chat 数据”，而是逐条核对：

1. 从当前 `tau_bench/envs/user.py` 的 AST 直接提取 `LLMUserSimulationEnv.build_system_prompt()`；
2. 1,513/1,513 system prompt 字符串精确相等；
3. 1,513/1,513 `messages[:-1]` 与 source case 的 runtime role-flipped prefix 精确相等；
4. target 始终是最后一个 `assistant` turn；
5. target 只有一句自然 Customer 回复或精确 `###STOP###`；
6. 无 `tool` role，角色严格交替；
7. Qwen3 训练和 vLLM runtime 都设置 `enable_thinking=false`；
8. `loss_mask_mode=last_assistant`，历史 Customer turns 只作上下文，不计算 loss。

Qwen3 的一个实现细节也已修正：`add_generation_prompt=True` 在关闭 thinking 时仍可能插入空 think block，直接用 prefix token 长度可能越过真实 target 起点。当前实现把“有 target 的最后 assistant render”和“同一 assistant content 置空的 render”做 token-level common prefix，只监督 target 内容与 assistant end-of-turn suffix。短 `###STOP###` 不再因旧的“至少 5 个 label token”门槛被误删。

## 7. Train / eval / TRL 导出合同

trial07 的 168 个自然状态全部进入 eval，其余 trial 进入 train；case ID 交集为 0。只在 train 对 terminal boundary 做 3 倍 materialized oversampling：

| 文件 | 行数 | CONTINUE | STOP | 分布 |
|---|---:|---:|---:|---|
| natural source | 1,513 | 1,313 | 200 | 原始 |
| rich / TRL train | 1,693 | 1,171 | 522 | train STOP ×3 |
| rich / TRL eval | 168 | 142 | 26 | 自然分布，不重采样 |

TRL 文件 1,861/1,861 行都只有一个顶层字段 `messages`。rich 文件保留 task ID、case ID、termination metadata、sample replica 和 provenance，供当前项目训练器、审计和错误分析使用。

这里的 trial07 eval 是 seen-task state-level model-selection split，不是业务泛化结论；同一 task instruction 可能在 train 中出现。10 个 unseen task 的 420 个状态从未发给 Teacher，也不应用作带 Teacher label 的 SFT eval，最终泛化必须通过微调后在线 rollout 单独评估。

## 8. 最终静态/数据验证结果

按“不要运行测试/训练”的要求，没有执行 pytest、LoRA 或 rollout；只运行了语法编译、diff whitespace、生成器自身门禁和只读数据合同检查。最终检查结果：

```json
{
  "validation": "passed",
  "runtime_prompt_exact_matches": 1513,
  "natural_cases": 1513,
  "raw_deepseek_export_gate": "passed",
  "audited_export_gate": "passed",
  "holdout_teacher_requests": 0,
  "privileged_leaks": 0,
  "train_rows": 1693,
  "eval_rows": 168,
  "trl_schema_only_messages": 1861,
  "secret_matches": 0
}
```

当前可以下的结论是：全量 seen User Simulator SFT 数据、runtime prompt、role contract、STOP 语义、TRL schema 与训练 loss mask 已对齐并可进入 LoRA。当前不能下的结论是“微调后 STOP latency 或主 Agent reward 已提升”；这必须由后续 frozen-policy A/B、cross-simulator evaluation 和主 Policy RL 实验给出。
