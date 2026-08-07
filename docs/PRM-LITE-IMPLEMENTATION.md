# PRM-Lite 实现详解：从 v4 到 v6.1 的完整技术文档

> **文档目标**：解释 PRM-Lite 是什么、v4 的问题在哪里、v5/v6/v6.1 怎么改的、代码怎么读。
> **维护者**：每次改 scoring 规则前先改本文档。

---

## 目录

1. [PRM-Lite 是什么](#1-prm-lite-是什么)
2. [代码入口与数据流](#2-代码入口与数据流)
3. [v4 的完整实现](#3-v4-的完整实现)
4. [v4 的问题分析](#4-v4-的问题分析)
5. [v5 → v6 → v6.1 改动清单](#5-v5--v6--v61-改动清单)
6. [代码对照表：v4 vs v5](#6-代码对照表v4-vs-v5)
7. [失败→恢复案例：305 美元事故](#7-失败→恢复案例305-美元事故)
8. [如何修改 PRM-Lite](#8-如何修改-prm-lite)
9. [配置与实验](#9-配置与实验)
10. [测试与验证](#10-测试与验证)
11. [附录 C：v6/v6.1 加固改动](#附录-cv6v61-加固改动2026-08-实验分析后)
12. [附录 D：DAPO 配置与 GRPO/DAPO 选择](#附录-d-dapo-配置与-grpodapo-选择)

---

## 1. PRM-Lite 是什么

### 1.1 背景问题

在 τ-bench 航空订票任务里，Agent 需要多步工具调用（搜索航班 → 读取用户信息 → 预订）才能完成一次任务。终局奖励只有两个值：

- **成功**：`reward = 1.0`（数据库状态与 ground truth 一致）
- **失败**：`reward = 0.0`（任务未完成）

这叫 **稀疏终局奖励（sparse terminal reward）**。问题在于：模型只能知道"这次轨迹失败了"，但不知道**失败发生在哪里**、**哪些步骤是对的**。

### 1.2 PRM-Lite 的思路

**Process Reward Model - Lite**（轻量级过程奖励模型）：用规则（而非训练出的神经网络）为每个 tool call 打分，叠加到终局奖励上，提供**密集的中间信号**。

公式：
```
最终得分 = outcome + α × process_score
```
- `outcome ∈ {0, 1}`：终局任务成功/失败
- `process_score ∈ [-0.5, 0.5]`：规则式过程评分（per-step 评分的均值）
- `α = 0.3`：过程信号的权重（由 tau_bench_interaction.py 的 `_compute_prm_lite_reward` 硬编码）

### 1.3 与 Turn-PPO 的关系

```
Turn-PPO（严格模式）：
  token_level_rewards = outcome  (terminal only)
  → advantage 全部由 critic + GAE 产生

PRM-Lite + Turn-PPO（实验模式）：
  token_level_rewards = outcome + 0.3 × process_score  (per turn)
  → 部分 advantage 由规则信号产生，critic 补充剩余部分
```

**当前默认配置**（`prm_lite.yaml`）：`algorithm.adv_estimator=grpo`，即 PRM-Lite 本身就是 advantage 来源，不走 critic。

---

## 2. 代码入口与数据流

### 2.1 关键文件

| 文件 | 职责 |
|---|---|
| `src/envs/tau_bench_interaction.py` | **核心评分逻辑**，`_compute_reasoning_quality_score`（v5）在此 |
| `src/envs/tau_bench_tools.py` | 14 个 airline tool 的 `execute()`，写入 `action_history` |
| `src/envs/tau_bench_context.py` | `CURRENT_TAU_STATE`（contextvar），`make_initial_state()` |
| `configs/interaction_config/tau_bench_airline_prm_lite.yaml` | `reward_mode: prm_lite` |

### 2.2 数据流

```
tau_bench_tools.py Tool.execute()
    ↓  appends to state["action_history"]
        {
            "tool": "book_reservation",
            "parameters": {...},
            "param_str": "...",
            "inc_reward": 0.0,        # tau_bench 内部增量奖励
            "is_error": True,         # obs.startswith("Error:")
            "observation": "...",     # 工具返回值（v5 新增）
            "extracted_entities": {}, # 正则提取的关键实体
            "content": "...",         # assistant 原始文本
        }

tau_bench_interaction.py _compute_reasoning_quality_score(action_history)
    ↓  returns process_score ∈ [-0.5, 0.5]
        _compute_prm_lite_reward(state)
            ↓  returns outcome + 0.3 × process_score
```

### 2.3 `action_history` 的生命周期

`action_history` 在以下三个地方被写入：

| 写入点 | 工具 | 内容 |
|---|---|---|
| `tau_bench_tools.py` `execute()` | 14 个 airline tool | tool call 结果 |
| `tau_bench_tools.py` `execute()` | 异常捕获（`except`） | `record_policy_error_action()` |
| `tau_bench_interaction.py` `generate_response()` | `implicit_think` 检测 | assistant 纯文本回复（> 100 字符） |

---

## 3. v4 的完整实现

v4 的评分函数为 `_compute_reasoning_quality_score`，在 `tau_bench_interaction.py` 的 line ~230–410（v5 发布前是 ~230–360）。

### 3.1 工具分类常量

```python
_READ_TOOLS = frozenset({
    "list_all_airports", "search_direct_flight", "search_onestop_flight",
    "get_user_details", "get_reservation_details", "calculate",
})
_WRITE_TOOLS = frozenset({
    "book_reservation", "cancel_reservation",
    "update_reservation_baggages", "update_reservation_passengers",
    "update_reservation_flights", "send_certificate",
})
_ESCALATION_TOOLS = frozenset({"transfer_to_human_agents"})
_THINK_TOOLS = frozenset({"think", "implicit_think"})
```

### 3.2 评分维度一览

| 标签 | 名称 | v4 值 | 条件 |
|---|---|---|---|
| **P0** | 动作错误惩罚 | **-0.10** | `action["is_error"] == True`（字符串前缀检测） |
| **P1** | placeholder 参数惩罚 | -0.05 / -0.03 | 参数含 placeholder 关键词或格式不符 |
| **P2** | 冗余动作惩罚 | -0.03 | 前 3 步内出现过完全相同的 tool+param |
| **P3** | 错误后恢复奖励 | +0.05 / -0.04 | 前一步错误且参数相同→重复罚，参数不同→恢复奖 |
| **P4** | 转人工惩罚 | -0.10 / -0.05 | 未读取数据就转人工→重罚 |
| **P9** | 廉价推理惩罚 | -0.02 | assistant content 长度 0 < len < 30 |
| **B1** | 数据链奖励 | +0.08 / +0.04 | 本步参数复用了上步提取的实体 |
| **B2** | 首次读取探索奖励 | +0.01 | 该工具首次出现（read 工具集内） |
| **B4/B5** | think 奖励 | +0.01 | think 后紧接非 placeholder 工具（1 步检测） |
| **P5** | 无推理惩罚 | **-0.05** | 轨迹 ≥ 3 步且没有任何 think |
| **B7** | 读取多样性奖励 | +0.01 | 读取工具种类 ≥ **3** |
| **P8** | 长度惩罚 | -0.01/步 | 工具调用数 > **8** |

### 3.3 v4 的核心问题（6 条）

| # | 问题 | 影响 |
|---|---|---|
| **Bug 1** | `is_error` 用 `obs.startswith("Error:")` 字符串检测，漏掉"软失败"（inc_reward=0 但 obs 无 Error 前缀） | P0 惩罚失效，假阳性低 reward |
| **Bug 2** | `implicit_think` 被重复记入 `action_history`（`generate_response` 里防重逻辑有漏洞） | 同一个 think 被记两次，think count 虚高，P5 误判 |
| **Bug 3** | B4/B5 只检测 `action[i+1]`，无法防御 think→think→placeholder→tool（两段 think） | 模型可以 think 两次绕过检查 |
| **Bug 4** | B7 阈值 3：airline 实际只有 3 类 read 工具，触发门槛低 | 多样性奖励难以获得 |
| **Bug 5** | P8 阈值 8：airline 优质轨迹集中在 4-6 步，阈值过高 | 长轨迹惩罚几乎永不触发 |
| **Bug 6** | P3 不区分"同工具换参数"（修正）和"完全换工具"（改策略） | 修正行为得到 +0.05，与换策略同分，不够细粒度 |

---

## 4. v4 的问题分析

### 4.1 `is_error` 字符串检测的脆弱性

τ-bench 的 tool 返回有三种：

```python
# 硬失败（有 Error 前缀）
obs = "Error: payment amount does not add up, total price is 255, but paid 305"

# 软失败（无 Error 前缀，但 inc_reward=0）
obs = "payment amount does not add up, total price is 255, but paid 305"

# 成功（inc_reward > 0）
obs = '{"reservation_id": "HATHAT", ...}'
```

v4 的 `is_error` 只看第一种，导致软失败漏检。

### 4.2 `implicit_think` 双重记录

`tau_bench_interaction.py` 的 `generate_response()` 里：
```python
# 有防重，但只在 last_action 是 implicit_think 时检查
if last_action.get("tool") == "implicit_think" and ...content 重复...:
    pass  # 不记
else:
    state["action_history"].append({"tool": "implicit_think", ...})
```
若上一次记录的 `implicit_think` 内容为 "I should check..."，而本次 assistant content 是 "Let me first..."，两次都超过 100 字符 → 都记入 → think count × 2 → P5 误判。

### 4.3 B4/B5 的 1 步检测漏洞

```
[think] → [think] → [book_reservation]  ← v4 给 +0.01（只看到 action[i+1]）
                              ↑
                        这其实是绕过，think×2 说明没有真实推理
```

---

## 5. v5 → v6 → v6.1 改动清单

### 5.1 v5 改动（基于 τ-bench airline 轨迹分析）

| # | 改动 | v4 | v5 | 动机 |
|---|---|---|---|---|
| **1** | P0 错误检测 | `is_error`（字符串） | `inc_reward==0 and not _obs_indicates_success()`（语义） | 覆盖软失败 |
| **2** | implicit_think 奖励 | +0.03（已被 v4-optimal 移除） | **0**（保持移除） | 防止双倍计数 |
| **3** | B4/B5 anti-hacking | 1 步检测 | **2 步检测**（action[i+1] + action[i+2]） | 防御 think-think 绕过 |
| **4** | B7 读取多样性阈值 | ≥ 3 种 | ≥ **2** 种 | airline 实际只有 3 类 |
| **5** | P8 长度惩罚阈值 | > 8 步 | > **6** 步 | 优质轨迹 4-6 步 |
| **6** | P8 每步惩罚幅度 | -0.01 | **-0.005** | 减轻惩罚强度 |
| **7** | P3 recovery 奖励 | 同工具不同参数 +0.05 | **+0.02**（更保守） | 区分修正 vs 换策略 |
| **8** | P5 无推理惩罚 | ≥ 3 步 → -0.05 | ≥ **3 步 → -0.05**；≥ **2 步 → -0.02** | 两步轨迹也有弱惩罚 |
| **9** | P9 cheap reasoning | content 长度 < 30 → -0.02 | **不变** | 已合理 |

### 5.2 v6 加固（基于实验现象分析，2026-08）

| # | 改动 | v5 | v6 | 动机（实验现象） |
|---|---|---|---|---|
| **A3** | 无效工具惩罚 | 无 | **-0.05/次** | 37 次 Unknown tool 调用，成功轨迹 reward 仍 0.9792 |
| **A4** | 递归 placeholder 检查 | 仅顶层 | **递归 flatten** | 占位值嵌套在 flights[]/passengers[] 中 |
| **P3-v6** | recovery bonus 条件化 | 任意错误后均可 | **仅合法后端错误**后 | 无效工具后切任意工具也拿 +0.03 bonus |

### 5.3 v6.1 长度放宽（2026-08 第二轮）

| # | 改动 | v6 | v6.1 | 动机 |
|---|---|---|---|---|
| **P8** | 长度惩罚阈值 | 6 | **10** | 复杂预订流程常有 8-12 步合法交互 |
| **P8** | 每步惩罚幅度 | -0.005 | **-0.003** | 降低误罚强度 |

---

## 6. 代码对照表：v4 vs v5

> 以下对比基于 `tau_bench_interaction.py` 中的 `_compute_reasoning_quality_score` 函数。

### 6.1 P0 核心判定

**v4**（line ~260）：
```python
if action.get("is_error", False):   # 字符串前缀检测
    score += _PRM_LITE_ACTION_ERROR_PENALTY  # -0.10
```

**v5**（line ~255–265）：
```python
is_failure = _is_tool_call_failure(
    float(action.get("inc_reward", 0.0)),
    action.get("observation", ""),
    tool,
)
if is_failure:
    score += _PRM_LITE_ACTION_ERROR_PENALTY  # -0.10
```
v5 新增辅助函数：
```python
def _obs_indicates_success(obs: str, tool_name: str) -> bool:
    """非Error的返回值中，有实质进展（HAT/bracket/detail等）表示成功。"""
    ...

def _is_tool_call_failure(inc_reward: float, obs: str, tool_name: str) -> bool:
    """inc_reward==0 且 obs 无成功指示词 → 软失败"""
    return bool(inc_reward == 0.0 and not _obs_indicates_success(obs, tool_name))
```

### 6.2 P3 错误恢复判定

**v4**（line ~280–290）：
```python
if i >= 1 and tool not in _THINK_TOOLS:
    prev = action_history[i - 1]
    if prev.get("is_error", False):
        prev_sig = (prev.get("tool", ""), prev.get("param_str", ""))
        curr_sig = (tool, pstr)
        if curr_sig == prev_sig:
            score -= 0.04  # 重复错误
        else:
            score += 0.05  # 不同参数 → +0.05（不管是修正还是换工具）
```

**v5**（line ~275–300）：
```python
if i >= 1 and tool not in _THINK_TOOLS:
    prev = action_history[i - 1]
    # v5: 同样用 inc_reward 判定前一步是否失败
    prev_is_failure = bool(
        prev.get("is_error", False)
        or _is_tool_call_failure(float(prev.get("inc_reward", 0.0)), ...)
    )
    if prev_is_failure:
        curr_sig = (tool, pstr)
        if curr_sig == prev_sig:
            score -= 0.04          # 重复同一个错误
        elif tool != prev["tool"]:
            score += 0.03          # 换工具（策略改变）
        else:
            score += 0.02          # 同工具换参数 → 修正行为（+0.02，比 v4 的 +0.05 更保守）
```

### 6.3 B4/B5 think anti-hacking

**v4**（line ~330–360）：
```python
# 只检查 action[i+1]
if next1_tool in _THINK_TOOLS:
    pass  # think → think → 不奖励
elif _has_placeholder(next1_params) or _is_redundant(...):
    pass  # 被绕过
else:
    score += 0.01
```

**v5**（line ~330–365）：
```python
# 扩展到 action[i+2]
if next1_tool in _THINK_TOOLS:
    if i + 2 < len(action_history):
        next2 = action_history[i + 2]
        if next2_tool not in _THINK_TOOLS and not _has_placeholder(next2_params):
            safe = True  # think → think → 非placeholder → 真实工具
    ...
elif _has_placeholder(next1_params) or _is_redundant(...):
    pass
else:
    safe = True  # think → 直接非placeholder工具
```

### 6.4 轨迹级调整

**v4**（line ~380–410）：
```python
# P5: >= 3 步无 think → -0.05
if think_count == 0 and len(action_history) >= 3:
    mean_score -= 0.05

# B7: >= 3 种 read 工具
if len(all_reads) >= 3:
    mean_score += 0.01

# P8: > 8 步，-0.01/步
length_threshold = 8
length_penalty_per_step = -0.01
```

**v5**（line ~380–410）：
```python
# P5: >= 3 步无 think → -0.05；>= 2 步无 think → -0.02
if think_count == 0:
    if len(action_history) >= 3:
        mean_score -= 0.05
    elif len(action_history) >= 2:
        mean_score -= 0.02  # 新增

# B7: >= 2 种 read 工具（airline 只有 3 类）
if len(all_reads) >= 2:
    mean_score += 0.01

# P8: > 6 步，-0.005/步（更轻）
length_threshold = 6
length_penalty_per_step = -0.005
```

### 6.5 动作历史字段更新

**v4**（`tau_bench_tools.py` line ~140）：
```python
state["action_history"].append({
    "tool": self.name,
    "parameters": parameters,
    "param_str": ...,
    "inc_reward": inc_reward,
    "done": is_done,
    "is_error": bool(obs and obs.startswith("Error:")),
    "extracted_entities": _extract_entities(obs),
    "content": assistant_content or "",
    # v4 没有 "observation" 字段
})
```

**v5**（`tau_bench_tools.py` line ~140）：
```python
state["action_history"].append({
    ...
    "is_error": bool(obs and obs.startswith("Error:")),  # 保留（向后兼容）
    "observation": obs,  # ★ v5 新增：PRM-Lite scoring 需要原始 obs 做软失败检测
    "extracted_entities": _extract_entities(obs),
    "content": assistant_content or "",
})
```

---

## 7. 失败→恢复案例：305 美元事故

### 7.1 场景（中文）

用户请求预订 5 月 20 日 JFK→SEA 经济舱，要求使用 250 美元代金券 + 5 美元信用卡尾号支付（代金券最多抵 250 美元，不能全抵）。

模型在 Turn 5 调用 `book_reservation`，参数写错了：
```python
payment_methods = [
    {"payment_id": "credit_card_4421486", "amount": 305}  # 错：应该用代金券抵 250 + 卡付 5
]
```
工具返回错误：
```
Error: payment amount does not add up, total price is 255, but paid 305
```

模型在 Turn 6 修正：
```python
payment_methods = [
    {"payment_id": "certificate_7504069", "amount": 250},  # 代金券
    {"payment_id": "credit_card_4421486", "amount": 5},     # 正确：卡只付 5
]
```
预订成功，任务终局 reward = 1.0。

### 7.2 v4 vs v5 评分对比

| Turn | v4 单步得分 | v5 单步得分 | v4 解释 | v5 解释 |
|---|---|---|---|---|
| Turn 1 搜索直飞 | +0.01 | +0.01 | B2 首次读取 | 同 |
| Turn 2 搜索中转 | +0.01 | +0.01 | B2 首次读取 | 同 |
| Turn 3 读用户信息 | +0.01 | +0.01 | B2 首次读取 | 同 |
| Turn 4 错误预订 | **-0.10** | **-0.10** | P0（is_error=True） | P0（`_is_tool_call_failure`） |
| Turn 5 修正预订 | **+0.05** | **+0.02** | P3 不同参数→+0.05 | P3 同工具换参数→+0.02（保守） |
| **轨迹均值** | -0.02 | -0.03 | P5 -0.05 + B7 +0.01 | 同 |
| **最终 PRM** | -0.06 | -0.09 | — | v5 更保守 |

**核心差异**：v4 把"同工具换参数"的修正行为给 +0.05，v5 给 +0.02（保守）。因为同工具换参数可能是**修正**（好），也可能是**随机试错**（不好），v5 选择更保守。

### 7.3 对 Turn-PPO 的影响

即使这条轨迹最终成功，v4 的 PRM-Lite 总分更高（-0.06 vs -0.09），意味着 **v5 对"失败→修正"行为的内部评分更严格**。如果用 PRM-Lite 作为训练信号，v5 会让模型更倾向于"一次性做对"而不是"试错后修正"。

---

## 8. 如何修改 PRM-Lite

### 8.1 新增一个评分维度（以 P10 为例）

**步骤 1**：在工具分类区添加常量（如有需要）：
```python
_SPECIAL_TOOLS = frozenset({"book_reservation", "cancel_reservation"})
```

**步骤 2**：在 `_compute_reasoning_quality_score` 的 for 循环内添加判定：
```python
# 位置：每个 action 的 per_step_scores.append(score) 之前
# P10: 预订操作成功后发送确认消息 +0.02
if tool == "book_reservation":
    if action.get("inc_reward", 0) > 0:
        score += 0.02
```

**步骤 3**：在对应的测试文件（`test_prm_lite_v5.py`）添加用例：
```python
def test_p10_booking_confirmation():
    history = [
        make("book_reservation", inc_reward=1.0,
             observation="Reservation confirmed HATHAT")
    ]
    s = _compute_reasoning_quality_score(history)
    assert abs(s - 0.02) < 1e-3
```

**步骤 4**：运行测试：
```bash
cd agentic-grpo-longhorizon
python src/envs/tests/test_prm_lite_v5.py
```

### 8.2 调整惩罚幅度

**不要直接改数字**，而是先在 recovery case 上验证：
```bash
cd agentic-grpo-longhorizon
python scripts/test/turn_ppo_recovery_case.py
```
观察 v4/v5 差异是否在预期范围内。

### 8.3 避免的常见错误

| 错误 | 后果 |
|---|---|
| 用 `action.get("observations")`（复数） | 字段不存在时返回 `None`，error 检测全部失效 |
| `_param_str` 返回简写工具名 | 导致 P3 的签名比较失效，重复惩罚误触 |
| `implicit_think` 不做去重 | 同一个 think 被记两次，P5 误判 |
| 阈值不参考 airline 工具集 | B7 阈值 3 对 airline 来说太低（只有 3 类 read 工具） |

---

## 9. 配置与实验

### 9.1 PRM-Lite 训练配置

```yaml
# configs/train/grpo/prm_lite.yaml
algorithm:
  adv_estimator: grpo         # PRM-Lite 自己提供 advantage
  kl_ctrl:
    kl_coef: 0.01
    rollout_correction:
      bypass_mode: true

rollout:
  multi_turn:
    interaction_config_path: configs/interaction_config/tau_bench_airline_prm_lite.yaml
```

### 9.2 交互层配置

```yaml
# configs/interaction_config/tau_bench_airline_prm_lite.yaml
reward_mode: prm_lite    # 触发 _compute_prm_lite_reward
```

### 9.3 实验命名

| 实验 | 配置 | 描述 |
|---|---|---|
| Exp 3 | `prm_lite.yaml` | PRM-Lite v4，300 步 GRPO |
| Exp 4 | `prm_lite_lata.yaml` | PRM-Lite v4 + LATA advantage（混合方案） |

---

## 10. 测试与验证

### 10.1 单元测试

```bash
# v4 测试（对应 v4 的评分规则）
cd agentic-grpo-longhorizon
python src/envs/tests/test_prm_lite_v4.py

# v5 测试（对应 v5 的评分规则）
python src/envs/tests/test_prm_lite_v5.py
```

### 10.2 恢复案例验证

```bash
cd agentic-grpo-longhorizon
python scripts/test/turn_ppo_recovery_case.py
```
预期输出：v5 在 Turn 6（修正）得分低于 v4，证明 P3 的保守改进生效。

### 10.3 快速验证

```bash
cd agentic-grpo-longhorizon
python scripts/test/prm_lite_quick_validate.py
```

### 10.4 诊断工具

```bash
cd agentic-grpo-longhorizon
python scripts/test/diagnose_prm_lite.py
```

---

## 附录 A：评分维度速查表

| 维度 | 标签 | 类型 | v4 | v5 | 触发条件 |
|---|---|---|---|---|---|
| 动作错误 | P0 | 惩罚 | -0.10 | -0.10 | 工具调用失败（v4: 字符串前缀；v5: inc_reward==0 + 无成功指示词） |
| placeholder 参数 | P1 | 惩罚 | -0.05/-0.03 | 同 | 参数含 placeholder 关键词或格式不符 |
| 冗余动作 | P2 | 惩罚 | -0.03 | 同 | 前 3 步内完全相同的 tool+参数 |
| 错误恢复 | P3 | 奖励 | +0.05/-0.04 | +0.03/+0.02/-0.04 | 前一步失败后的动作 |
| 转人工 | P4 | 惩罚 | -0.10/-0.05 | 同 | 未经数据收集就转人工 |
| 无推理 | P5 | 惩罚 | -0.05 | -0.05/-0.02 | ≥ 3 步（v4）；≥ 2 步（v5 弱）/≥ 3 步（v5 强）|
| 长度惩罚 | P8 | 惩罚 | -0.01/步 | -0.005/步 | > 8 步（v4）；> 6 步（v5） |
| 廉价推理 | P9 | 惩罚 | -0.02 | 同 | assistant content 长度 0 < len < 30 |
| 数据链 | B1 | 奖励 | +0.08/+0.04 | 同 | 本步参数复用上步提取的实体 |
| 首次读取 | B2 | 奖励 | +0.01 | 同 | read 工具首次出现 |
| think 奖励 | B4/B5 | 奖励 | +0.01 | 同 | think 后接非 placeholder 工具 |
| 读取多样性 | B7 | 奖励 | +0.01 | +0.01 | ≥ 3 种（v4）；≥ 2 种（v5）read 工具 |

---

## 附录 B：关键代码位置

| 功能 | 文件 | 函数/行 |
|---|---|---|
| v5 评分核心 | `tau_bench_interaction.py` | `_compute_reasoning_quality_score` (~line 230) |
| v5 辅助：软失败检测 | `tau_bench_interaction.py` | `_is_tool_call_failure` (~line 220) |
| v5 辅助：成功指示词 | `tau_bench_interaction.py` | `_obs_indicates_success` (~line 190) |
| 最终奖励公式 | `tau_bench_interaction.py` | `_compute_prm_lite_reward` (~line 400) |
| action_history 写入 | `tau_bench_tools.py` | `execute()` (~line 140) |
| implicit_think 记录 | `tau_bench_interaction.py` | `generate_response()` (~line 710) |
| 交互配置开关 | `tau_bench_interaction.py` | `__init__()` (~line 580) |
| v4 独立版本（用于对比） | `scripts/test/turn_ppo_real_data_dryrun.py` | `_compute_reasoning_quality_score_v4` |
| v5 独立版本（带 breakdown） | `scripts/test/turn_ppo_real_data_dryrun.py` | `_compute_reasoning_quality_score_v5` |

---

## 附录 C：v6 加固改动（2026-08 实验分析后）

> **触发背景**：step 217 训练日志 + W&B 实验数据分析，发现 reward 信号统计不稳定，process score 均值接近 0（-0.0337），outcome 贡献占主导，过程启发式规则存在语义验证不足问题。

### C.1 三项加固改动

| # | 改动 | v5 | v6 | 动机（实验现象） |
|---|---|---|---|---|
| **A3** | 无效工具惩罚 | 无 | **-0.05** | 37 次 Unknown tool 调用，reward 仍接近成功（0.9792），模型未从"该工具不存在"信号中学习 |
| **A4** | 递归 placeholder 检查 | 仅顶层 | **递归 flatten**（flights[], passengers[]） | 占位值可能嵌套在列表对象中，原检查只扫顶层 |
| **P3-v6** | recovery bonus 条件化 | 任意错误后均可 | **仅合法后端错误**后恢复 | 无效工具后切换到任意工具也能拿 +0.03 recovery bonus，放大启发式行为 |

### C.2 关键代码变化

**A3：无效工具检测逻辑**

```python
# tau_bench_interaction.py — P1 之前插入
is_invalid_tool = (
    action.get("is_error", False)
    and tool not in _VALID_TOOLS
    and obs.startswith("Error: Unknown tool")  # tau-bench base.py else-branch
)
if is_invalid_tool:
    score -= 0.05
```

**A4：递归 placeholder 检查**

```python
# tau_bench_interaction.py — 新增 _flatten_params + 改写 _has_placeholder
def _flatten_params(params: dict) -> list[tuple[str, Any]]:
    result = []
    def _flatten(obj, prefix):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _flatten(item, f"{prefix}[{idx}]")
        else:
            result.append((prefix, obj))
    _flatten(params, "")
    return result

def _has_placeholder(params: dict) -> bool:
    for field_name, value in _flatten_params(params):  # 递归遍历
        if _is_placeholder_param(field_name, value):
            return True
    return False
```

**P3-v6：recovery bonus 条件化**

```python
# P3 块内：invalid tool 后不触发 recovery bonus
if prev.get("is_error", False):
    prev_is_invalid = (
        prev.get("tool", "") not in _VALID_TOOLS
        and prev.get("observation", "").startswith("Error: Unknown tool")
    )
    if prev_is_invalid:
        pass  # v6: no recovery bonus after invalid tool
    else:
        # ... 原有 P3 recovery logic ...
```

### C.3 合法工具白名单

```python
_VALID_TOOLS = frozenset({
    "book_reservation", "cancel_reservation", "update_reservation_baggages",
    "update_reservation_passengers", "update_reservation_flights", "send_certificate",
    "list_all_airports", "search_direct_flight", "search_onestop_flight",
    "get_user_details", "get_reservation_details", "calculate",
    "think", "implicit_think", "transfer_to_human_agents",
})
```

### C.4 metadata 扩展（W&B 可观测性）

`generate_response()` 最终返回的 metadata 新增字段：

```python
"outcome": outcome,           # 二值成功/失败（独立于 process）
"process_score": process_score,  # 原始过程分（未裁剪到 ±0.5）
"invalid_tool_count": invalid_count,  # 无效工具调用次数
```

> **注意**：这些字段不参与 GRPO 计算，仅用于 W&B 日志记录和分析。

### C.5 W&B 监控建议

实验时应同时记录和监控：

| 指标 | 含义 | 预期变化方向 |
|---|---|---|
| `outcome` | 二值成功率 | 稳定或上升 |
| `process_score` | 过程分均值 | 负偏改善（更少无效工具、占位符） |
| `invalid_tool_count` | 无效工具调用次数 | 下降 |
| `reward` | 总 reward | outcome 主导，process 贡献稳定化 |

### C.6 v6.1 长度惩罚放宽（2026-08 第二轮优化）

基于实验现象分析（step 217 训练数据），发现 `history > 6` 的轨迹往往是合理的复杂预订流程，而非冗余循环。

| 参数 | v5 | v6.1 | 原因 |
|---|---|---|---|
| `length_threshold` | 6 | **10** | airline 任务常有 8-12 步合法流程 |
| `length_penalty_per_step` | -0.005 | **-0.003** | 降低每次超长的惩罚强度 |

预期效果：减少对合法复杂任务的误罚，降低 process_score 负偏。

### C.7 工具 schema 描述优化

实验发现模型会调用不存在的工具（如 `search_flight_with_stops`）。改进 `tau_bench_airline_tools.yaml` 中的工具描述，明确工具边界：

- `book_reservation`: 补充必需字段说明（user_id, flight_type, cabin, flights, passengers, payment）
- `search_direct_flight`: 明确只搜索直飞，多程用 `search_onestop_flight`
- `search_onestop_flight`: 明确只搜索单程中转
- `get_reservation_details`: 明确 reservation_id 格式（6位字母数字）
- `get_user_details`: 明确 user_id 格式（first_last_number）
- `cancel_reservation`: 补充警告信息
- `send_certificate`: 补充使用场景和必需字段

> **注意**：schema 描述修改后需要重新部署工具配置才能生效。

### C.8 GRPO 方差问题调研（DAPO / GSPO）

**实验现象**：step 40/44/78 reward 跳变 ±3~8σ，step 210→211 下降 0.40。

**调研结论**：

| 算法 | 核心改进 | 适用性 | 实施难度 |
|---|---|---|---|
| **DAPO Dynamic Sampling** | 过滤 silent groups（std=0），确保每批都有有效梯度 | **直接相关** | 需改 verl 核心（`use_dynamic_sampling=True`） |
| **DAPO Clip-Higher** | 非对称裁剪（ε_low < ε_high），防熵崩溃 | 次相关 | 需改 verl 核心 |
| **DAPO Token-Level Loss** | 归一化所有 token 梯度，防长序列偏差 | 次相关 | 需改 verl 核心 |
| **GSPO** | Sequence-level importance ratio 替换 token-level | 适合 MoE + 长序列 | 需改 verl 核心，激进 |
| **当前 GRPO** | 标准实现 | — | — |

**核心判断**：当前 reward 跳变最可能是 **silent groups** 造成的（n=8 极小 group size，二值 outcome → 高概率全成功或全失败 → std=0 → 无梯度更新）。

**推荐方案（按优先级）**：

1. **近期可行**：增大 `rollout.n`（当前 n=8），每组更多样本 → silent group 概率降低
2. **DAPO Dynamic Sampling**：若 verl 版本支持，启用 `use_dynamic_sampling=True` 开关
3. **Task-level 归一化**：跨任务分组统计，不只看 step 内均值
4. **Momentum baseline**：用移动平均 reward 替代单步 reward 作为参考

**不推荐强加的方案**：
- 调整 `outcome + 0.3 * process` 权重（需 ablation）
- GSPO（改动过大，需 verl 核心重构）
- 盲目增大 group size 而不测吞吐

### C.9 未实施的建议（需进一步验证）

| 建议 | 原因 | 后续行动 |
|---|---|---|
| 调整 `outcome + 0.3 * process` 权重 | 需要 ablation 实验，不能随意动 | 建议按消融实验设计验证 |
| 固定长度惩罚任务条件化 | 需要任务基线信息 | 需改交互配置，增加 task-specific baseline |
| 增大 group size | 需改训练配置 | 需验证对训练吞吐的影响 |
| 拆分 JSON decode error | 属于模型输出协议问题 | 应与 prompt/prompt 修复协同 |

---

## 附录 D：DAPO 配置与 GRPO/DAPO 选择

> 基于 [DAPO 官方文档](https://github.com/volcengine/verl/blob/main/docs/algo/dapo.md) 和 verl v0.4+ 实现。

### D.1 什么时候需要 DAPO

当训练出现以下现象时，启用 DAPO Dynamic Sampling 可能有效：

| 现象 | 原因 | DAPO 是否有效 |
|---|---|---|
| step reward 跳变 ±3~8σ | n=8 小 group → 高概率 silent groups（全成功/全失败 → std=0 → 无梯度） | ✅ 直接解决 |
| reward 曲线剧烈抖动 | 同上 | ✅ 直接解决 |
| 训练到某 step 后突然发散 | silent groups 导致梯度消失，突然恢复时噪声大 | ✅ 缓解 |
| 正常稳定训练 | — | ⚠️ 不需要，可能增加采样时间 |

**诊断方法**：观察 W&B 中 `train/num_gen_batches` 指标。如果经常 >1，说明 silent groups 频繁出现。

### D.2 GRPO vs DAPO 配置选择

| 配置 | YAML | Trainer | 适用场景 |
|---|---|---|---|
| **PRM-Lite + GRPO**（当前基线） | `prm_lite.yaml` | `RayPPOTrainer` | 正常训练，快速实验 |
| **PRM-Lite + DAPO**（推荐对比） | `prm_lite_dapo.yaml` | `RayDAPOTrainer` | 方差问题显著时，精确实验 |

两个配置仅在 `algorithm.filter_groups` 和 `actor.clip_ratio_low/high` 处不同。

### D.3 核心机制解释

#### D3.1 Dynamic Sampling（解决 silent groups）

```
prompt 组 G = {s1, s2, ..., s8}  ← n=8 个样本

如果 all(score_i == 1.0) 或 all(score_i == 0.0)：
    → std(G) = 0
    → advantage = 0 for all
    → 无梯度更新，浪费一次采样

DAPO Dynamic Sampling：
    → 过滤 std=0 的组
    → 重新采样，直到有 variance
    → 保证每步训练都有有效梯度信号
```

在 verl 中，filter 用 `algorithm.filter_groups` 配置：
```yaml
algorithm:
  filter_groups:
    enable: True
    metric: seq_final_reward  # sum(reward) per trajectory
    max_num_gen_batches: 10   # 最多重采样 10 次
```

#### D3.2 Clip-Higher（防熵崩溃）

```
标准 PPO/GRPO：clip(ratio, 1-ε, 1+ε)  # ε_low = ε_high = 0.2

DAPO Clip-Higher：clip(ratio, 1-ε_low, 1+ε_high)  # ε_low=0.2, ε_high=0.28

效果：
    - 上界更宽松（0.28）→ 允许更多大幅度更新
    - 下界更紧（0.2）→ 防止过度保守
    → 防止策略熵快速下降
```

#### D3.3 Token-Level Loss（防长序列稀释）

```
长轨迹（12288 tokens）和短轨迹（1024 tokens）在 seq-mean 模式下：
    - 长轨迹有更多 token，每个 token 的梯度被稀释
    - 短轨迹每个 token 梯度权重更高

token-mean 模式：所有 token 等权平均
    → 长轨迹的更多 token 被充分学习
    → 适合 τ-bench 这类需要长 CoT 的任务
```

### D.4 启动方式

```bash
# GRPO（基线）
python -m verl.trainer.main_ppo \
    +config=train/grpo/prm_lite

# DAPO（对比实验）
python -m verl.trainer.main_ppo \
    +config=train/grpo/prm_lite_dapo
```

> 注意：`main_ppo.py` 会根据 `algorithm.filter_groups.enable` 自动选择 `RayDAPOTrainer` 或 `RayPPOTrainer`，无需手动指定 trainer 类。

### D.5 监控指标

| 指标 | 含义 | DAPO vs GRPO 预期差异 |
|---|---|---|
| `train/num_gen_batches` | 每步采样的批次数 | DAPO 应 > 1（说明在重采样） |
| `train/reward` 均值 | 平均 reward | 应更平稳 |
| `train/reward` std | reward 波动 | DAPO 应更小 |
| `train/entropy` | 策略熵 | DAPO 应下降更慢（Clip-Higher 效果） |
| `train/clipfrac` | 裁剪比例 | DAPO 可能更低 |
| `throughput/samples_per_sec` | 采样吞吐量 | DAPO 可能略低（重采样开销） |

### D.6 注意事项

1. **DAPO 需要 verl ≥ 0.4**（支持 `filter_groups` 配置）
2. **`max_num_gen_batches` 不要设太大**：避免在极难任务上无限重采样
3. **`loss_agg_mode: token-mean` 可能不适合所有任务**：对于简单任务（短输出），`seq-mean` 可能更合适
4. **DAPO 不解决 reward 设计问题**：如果 process score 语义不正确，DAPO 只让训练更稳定，不会改善效果
