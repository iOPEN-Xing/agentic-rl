# Agentic-GRPO 代码审核报告（修正版）

> **审核日期**：2026-07-30
> **代码仓库**：https://github.com/iOPEN-Xing/agentic-rl
> **审核分支**：
> - `codex/turn-level-reward` → 方案一
> - `codex/user-simulator-judge` → 方案二
> - `codex/hybrid-advantage` → 方案三
> **方案二代码配置**：Policy Qwen3-8B + LoRA r=16/α=32 | User Simulator / Judge：Qwen3-14B（硬件与权重加载以各分支代码为准）

---

## 1. 代码边界澄清

**重要修正**：原报告存在严重的文件归因错误。方案三的核心文件被错误标记为 `reward_fusion.py`，该文件实际只存在于方案二分支。以下是三个方案的真实代码边界：

| 方案 | 分支 | 核心文件 | 真实职责 |
|------|------|----------|----------|
| 方案一 | `codex/turn-level-reward` | `tau_bench_interaction.py` | PRM-Lite 规则评分 + Potential Shaping |
| 方案一 | `codex/turn-level-reward` | `verl/trainer/ppo/core_algos.py` | **标准 GRPO**（非 turn-level advantage） |
| 方案二 | `codex/user-simulator-judge` | `user_simulator_judge.py` | LLM-as-Judge 评估 |
| 方案二 | `codex/user-simulator-judge` | `reward_fusion.py` | Judge 中性化 + 跨轨迹一致性诊断（已移除 session 多源融合） |
| 方案三 | `codex/hybrid-advantage` | `verl/trainer/ppo/hybrid_advantage.py` | **Session GRPO + Turn MC Return 融合** |
| 方案三 | `codex/hybrid-advantage` | `verl/experimental/agent_loop/tool_agent_loop.py` | Assistant span 记录 + event reward 附加 |

---

## 2. 方案一审核：Turn-Level Reward Signal

### 2.1 核心实现

**文件**：`tau_bench_interaction.py`（方案一分支）

**Turn reward 计算公式**：

```python
# 实际公式（非原报告描述的简单叠加）
r_turn = (process_weight * process_score + shaping_weight * shaping) / weight_sum
r_turn = clamp(r_turn, -1.0, 1.0)
```

**Process Score 实际规则**：
- 非 error 工具：+0.20
- 使用前序实体：+0.25（B1）
- 首次 read 探索：+0.10（B2）
- 非 error write：+0.15（B3）
- Placeholder 参数：-0.20（P1）
- 冗余操作：-0.10（P2）
- 过早 escalation：-0.25（P4）
- 无 thinking（>3 步）：-0.15（P5）
- 过长轨迹（>8 步）：-0.10（P6）
- Cheap reasoning（<30 chars）：-0.15（P9）

**Potential Shaping**：

```python
def compute_potential_shaping(previous_potential, current_potential, gamma=0.99):
    return gamma * current_potential - previous_potential
```

其中 potential 包含：
- 0.30 × unique_reads（最多 4）
- 0.30 × successful_writes（最多 2）
- 0.20 × chained_actions（最多 2）
- **0.20 × outcome**（最终是否成功）

### 2.2 审核意见

#### ⚠️ 核心问题：最终仍被压成 trajectory advantage

方案一的配置使用标准 GRPO：

```yaml
# configs/train/grpo/turn_level_reward.yaml:84
algorithm:
  adv_estimator: grpo
```

而标准 GRPO 的核心逻辑：

```python
# verl/trainer/ppo/core_algos.py:302
scores = token_level_rewards.sum(dim=-1)  # 全部 token reward 求和
# ... 组内归一化 ...
scores = scores.unsqueeze(-1) * response_mask  # 同一标量广播给所有 token
```

**结论**：虽然方案一在 `tau_bench_interaction.py` 中计算了 turn-level reward 并对齐到 token，但下游 GRPO 仍将其压缩为每个 trajectory 一个标量。早期正确 turn 和后期错误 turn 获得相同 advantage。

这不是 Turn-PPO 实现。Turn-PPO 需要：
- 可学习的 turn-level value head
- Turn boundary 上的 GAE
- Turn-level PPO ratio

#### ⚠️ PBRS "策略不变性" 声明不成立

函数形式是标准的 potential shaping：
```python
F_t = γΦ(s_{t+1}) - Φ(s_t)
```

但下游链路存在三个破坏条件：

1. **Discount mismatch**：shaping 使用 γ=0.99，但 GRPO 对 token rewards 做未折扣求和
2. **Terminal potential 非零**：成功/失败轨迹的 terminal potential 不同
3. **Event count 归一化**：`ray_trainer.py:184` 按事件数量再次平均

仅删除 `0.20 * outcome` 项不能恢复 policy invariance。

#### ✅ 有价值的基础设施

- Action cursor 防止重复统计
- Turn span 记录
- Reward tensor 传递机制

---

## 3. 方案二审核：User Simulator as Judge

### 3.1 核心实现

**文件**：`user_simulator_judge.py` + `reward_fusion.py` + `tau_bench_interaction.py`

**Judge 输入**：
- task_goal（任务目标，前 4000 chars）
- latest_agent_message（当前回复，前 4000 chars）
- recent_conversation（最近 6 条对话）
- current_cycle_tool_trace（自上次用户回复以来的工具动作，包含 parameters、**is_error** 与截断 observation/error reason）

**审核后的评分语义**：
```python
raw_score = 0.45 * task_progress + 0.40 * tool_correctness + 0.15 * communication
turn_signal = 2 * (raw_score - 0.5)  # neutral=0.5 时映射至 [-1, 1]
session_score = environment_outcome  # 严格保持 τ-bench 规则终局奖励
```

默认配置 `algorithm.judge_turn_reward.enabled=false`：采集 Judge 诊断但不改变 GRPO。只有离线校准通过后才建议以 `turn_weight=0.05` 做小权重消融。

### 3.2 审核意见

#### ⚠️ 仍存在：同 simulator 生成和评分

User simulator 与 judge 使用同一模型、同一 endpoint。这会产生相关偏差：
- Simulator 的对话偏好直接成为评分偏好
- Judge 不等于独立监督源

#### ✅ 已修复：证据重复计入

旧实现每轮融合 PRM/Judge，session 结束时再次融合，trainer 又叠加 turn contribution。当前实现已移除 session 多源融合和 PRM 重复路径：terminal score 只取 outcome，Judge 仅保留一个可关闭的 interaction-cycle auxiliary path。

#### ✅ 已修复：Judge 失败改变奖励标尺

```python
# invalid / timeout / parse failure
return None  # 不产生 event，turn_event_mask 不置位
```

Judge 失败不再重归一化其他来源，也不再给一个伪造的 0 分 event；仍需监控 valid rate，防止选择性缺失。

#### ✅ 已补充：完整工具结果证据

原实现只有 `is_error` 布尔值。当前 `tau_bench_tools.py` 会记录截断后的 observation；env exception 也写入 action history。Judge prompt 明确要求以 tool observation 为准，不得相信 assistant 自称成功。终局 tool correctness 仍只由环境 verifier 决定。

#### ⚠️ 粒度边界

当前 Judge 并非逐个 tool turn 调用，而是在用户回复前评价一个 interaction cycle（可能包含多个工具动作 + 最终 assistant message）。辅助信号虽然对齐到该段最后一个可训练 token，但当前 `adv_estimator=grpo` 会先对整条序列 reward 求和，再把同一个 trajectory advantage 广播给所有 response token。因此 span 位置只是记账，不产生 turn-specific credit，不能宣称实现了 UserRL 的逐 Turn token advantage。

#### ✅ 适合方向

- 默认 diagnostic-only：valid rate、成功/失败分布、跨 trajectory outcome correlation、冲突 case
- 校准通过后以 0.02/0.05 做小权重消融
- 不可验证维度（沟通质量、用户体验）评价
- 若要复现 UserRL，应另行实现规则化 dense turn reward + `grpo_multiturn`

---

## 4. 方案三审核：Hybrid Advantage（核心更正）

### 4.1 真实核心文件

**文件**：`verl/trainer/ppo/hybrid_advantage.py`

这是原报告最严重的错误——把 `reward_fusion.py`（属于方案二）列为方案三的核心文件。

### 4.2 真实实现逻辑

**不是**：Multi-Source Reward Fusion（方案二旧实现已在本轮复核中移除）

**是**：**Hybrid Session/Turn Monte-Carlo Advantage**

```python
# hybrid_advantage.py:176
def compute_grpo_hybrid_advantage(...):
    # Session advantage：标准 GRPO outcome advantage
    session_advantages, _ = compute_grpo_outcome_advantage(...)

    # Turn advantage：MC reward-to-go，按 assistant span 对齐
    turn_advantages, turn_mask = compute_turn_level_mc_advantage(...)

    # 融合
    final_advantages = α * session_advantages + (1-α) * turn_advantages
```

**Turn MC Return 计算**：

```python
# hybrid_advantage.py:130
reward_to_go = [0.0] * len(parsed_rewards)
running_return = 0.0
for turn_index in range(len(parsed_rewards) - 1, -1, -1):
    running_return = parsed_rewards[turn_index] + gamma * running_return
    reward_to_go[turn_index] = running_return
```

**关键**：assistant turn span 记录：

```python
# tool_agent_loop.py:274
turn_start = len(agent_data.response_mask)
append_assistant_turn(
    agent_data.assistant_turn_spans,
    agent_data.assistant_turn_rewards,
    turn_start,
    len(agent_data.response_mask),
)
```

### 4.3 审核意见

#### ✅ 三者中唯一真正形成 turn-specific advantage

- 记录了真实 assistant span
- Event reward 附到对应 action 上
- Turn MC return 是 critic-free 的，但方向正确

#### ⚠️ Turn MC return 本质是终端 reward 的位置折扣

由于 τ-bench 只有 terminal reward：
```python
G_t = γ^(T-t) * R_terminal
```

这是合理的 credit assignment，但不是真正学到的过程 advantage。

#### ⚠️ 同 turn_index 不代表同一状态

```python
# hybrid_advantage.py:136
grouped_turns[(uid, turn_index)].append(...)
```

同位置 turn 在不同轨迹中可能对应完全不同的环境状态。归一化依赖位置条件 baseline，不是状态条件 advantage。加上 `len(records) < 2: continue` 的 survivorship bias。

#### ⚠️ Schedule 后期 80% 依赖 turn signal

```python
# config: session_weight_start=0.8, session_weight_end=0.2
# → 训练末期 80% 依赖 turn signal
```

考虑到 turn signal 本质是终端 reward 的折扣，这个 schedule 过于激进。

#### ⚠️ Advantage 二次归一化问题

```python
# hybrid_advantage.py:216-217
session_advantages = normalize_advantage_component(session_advantages, response_mask, epsilon)
turn_advantages = normalize_advantage_component(turn_advantages, turn_mask, epsilon)
```

广播后若再做 token-weighted 归一化，会重新耦合不同 task 和 turn。

### 4.4 关于 "TRACE"

原报告说"方案三缺少 TRACE 实现"是错误的：

1. **方案三没有声称自己是 TRACE**。它实现的是 Hybrid MC Advantage
2. **TRACE 需要 gold-answer log-probability state value**，τ-bench 没有可直接使用的唯一 gold textual target
3. TRACE 应作为独立研究方向，不是现有 Hybrid 的代码缺陷

---

## 5. 共同问题（P0 优先于算法选择）

### 5.1 初始用户 observation 被丢弃

**训练**（`build_grpo_parquet.py:46`）：
```python
"prompt": [{"role": "system", "content": SYSTEM_PROMPT}]  # 只有 system
```

**评测**（`tau_bench_wrapper.py:140`）：
```python
{"role": "user", "content": str(obs_res.observation)}  # 正确包含 user
```

训练时 policy 在不知道用户任务的情况下先生成，评测却以 user message 开始。

### 5.2 `finalize_interaction()` 从未被调用

在 `tool_agent_loop.py` 中搜索 "finalize" 返回零匹配。`_instance_dict` 不会清理。

### 5.3 终止契约不完整

`tau_bench_tools.py:124` 设置 `state["done"] = True`，但 `tool_agent_loop.py` 没有消费此状态。可能对已终止环境执行 RESPOND。

### 5.4 `pass^4` 稳定性公式错误

使用 HumanEval 的 pass@k 公式（至少一次成功），而非 "4 次全部成功"。

---

## 6. 综合评级与建议

### 6.1 修正后的方案评价

| 方案 | 原报告评级 | 修正后评级 | 原因 |
|------|-----------|-----------|------|
| 方案一 | 推荐实施 | 暂不进入主训练 | Turn reward 最终被标准 GRPO 压成 trajectory advantage；PBRS 条件不成立 |
| 方案二 | 可选实施 | 诊断默认 / 小权重可消融 | terminal outcome 与 Judge 已解耦；同 simulator 偏差仍需校准 |
| **方案三** | **需要重构** | **三者最适合做主干** | **唯一真正形成 turn-specific advantage** |

### 6.2 正确优先级

#### P0：先修共同交互契约

1. 训练 parquet 包含初始 user observation
2. 训练与评测使用相同初始状态协议
3. `finalize_interaction()` 在所有路径被调用
4. `done` 正确终止 agent loop
5. Error 分类（policy failure / simulator error / infrastructure error）
6. 修正 `pass^4` 稳定性公式

#### P1：Hybrid 主线修正

1. 明确标注 `(uid, turn_index)` 为位置 baseline，不宣称 state advantage
2. 处理 survivorship bias（提前终止不进入后续组）
3. Turn/event 粒度归一化后广播，不再 token-weighted 二次归一化
4. **先使用固定小 turn 权重**（建议初始 α_session ≥ 0.8）
5. 环境 outcome 保持为主目标

#### P2：辅助研究方向

1. **Potential shaping**：如保留，需整体修复 terminal boundary、discount 和下游聚合，不是只删 outcome 项
2. **Judge**：已接入 batch/validation correlation；下一步做独立 Judge、分桶校准与冲突 case 审核
3. **TRACE**：定义清楚 gold target 表示后独立实现
4. **Turn critic**：MC return 稳定后引入 turn GAE

---

## 7. 最终结论

> **不要把"删除 terminal outcome + 给 Hybrid 补 TRACE"作为下一步。**
>
> **正确的下一步是先修共同交互契约，然后保留 Hybrid 的真实 turn span 与 MC reward-to-go，将它收敛成一个小权重、event-level normalization 的可靠 critic-free baseline；随后再决定是加入 turn critic，还是另起 TRACE 研究分支。**

原报告因文件归因错误（把 `reward_fusion.py` 误认为方案三核心）导致最终优先级几乎完全颠倒。

---

## 8. 附录：关键代码路径（修正）

| 分支 | 文件 | 关键函数/类 |
|------|------|-------------|
| 方案一 | `tau_bench_interaction.py` | `_compute_turn_reward()`, `compute_potential_shaping()` |
| 方案一 | `core_algos.py` | `compute_grpo_outcome_advantage()` ← **标准 GRPO** |
| 方案二 | `user_simulator_judge.py` | `UserSimulatorTurnJudge` |
| 方案二 | `reward_fusion.py` | `JudgeSignalPolicy`, `reward_consistency()` |
| 方案二 | `tau_bench_interaction.py` | interaction-cycle Judge 接线、pure outcome session score |
| **方案三** | **`hybrid_advantage.py`** | `compute_grpo_hybrid_advantage()`, `compute_turn_level_mc_advantage()` |
| 方案三 | `tool_agent_loop.py` | `append_assistant_turn()`（span 记录） |

---

*审核人：Claude Code Reviewer*
*审核版本：v2.0（修正版）*
*修正内容：文件归因错误、PBRS 分析不完整、方案优先级颠倒*

> 2026-07-30 方案二二次复核：Codex 对照 UserRL 论文、官方源码与
> `codex/user-simulator-judge` 分支后，更新了奖励解耦、interaction-cycle
> 粒度、工具 observation、默认诊断模式与跨轨迹监控结论。
