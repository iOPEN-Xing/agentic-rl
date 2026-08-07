# Agentic-GRPO-LongHorizon

> **用 PRM-Lite v5 解决多轮多工具 Agent 的 GRPO 训练崩溃**

---

## 问题本质：为什么多轮 Agent 上 vanilla GRPO 会崩溃？

τ-bench airline 每个任务最多 30 轮交互，模型需调用 14 个工具。vanilla GRPO 的三大问题在这里被极度放大：

**根因 1：组奖励饱和（Group Saturation）**
- `group_size = 8`，`reward_mode = binary`（成功=1.0，失败=0.0）
- 8 条轨迹中全是 0 或全是 1 → 组内 advantage 方差 = 0 → 梯度消失

**根因 2：训练集泄露偏差**
- 40 个训练任务中，16 个是 `covered_seen`（有 72B 教师轨迹）
- 策略直接记忆模式，`uncovered_seen` 和 `unseen` 任务上 pass ≈ 0

**根因 3：长度归一化惩罚推理**
- GRPO 用 `1/L` 归一化 response 长度
- 长思考轨迹被惩罚，策略学会"短思考 + 频繁试错" → step 150 后崩溃

---

## PRM-Lite v5 解决方案

PRM-Lite = Process Reward Model（Lite = 零训练参数，纯规则）。

给每个工具调用打分，叠加到 terminal reward 上：
```
reward = outcome + 0.3 × process_score
process_score ∈ [-0.5, +0.5]  # 所有 step 分数的均值
```

| 根因 | PRM-Lite v5 对应机制 |
|------|----------------------|
| 组饱和 | process_score 提供组内差异信号 |
| 训练集泄露 | placeholder 检测、错误恢复等规则对 unseen 同样有效 |
| 长度惩罚 | 合理评分鼓励有效推理，而非简单惩罚长度 |

---

## PRM-Lite v5 的 6 项优化（v4→v5）

### P0：软失败检测（最关键）

**v4 问题**：只检查 `obs.startswith("Error:")` 前缀，漏掉约 70% 的失败案例：

```python
# v4：硬编码前缀检测
is_error = obs and obs.startswith("Error:")

# v5：语义检测
_SOFT_FAIL_KEYWORDS = ["no flights found", "no reservation", "not found",
                       "could not find", "unable to find", "no available"]

def _obs_indicates_success(obs):
    if not obs or obs.startswith("Error:"):
        return False
    return any(kw in obs.lower() for kw in _SOFT_FAIL_KEYWORDS)

is_soft_fail = (
    inc_reward == 0
    and not action.get("is_error", False)
    and _obs_indicates_success(obs)
)
if is_soft_fail:
    score -= 0.02
    if i + 1 < len(action_history) and next_params:
        score += 0.02  # 恢复奖励
```

**工程逻辑**：airline 中"搜索无结果"、"未找到订单"是最常见的失败形式，v4 靠前缀匹配覆盖率仅 ~30%。

---

### B4/B5：Think 反越狱扩展到 2 步

**v4 问题**：只检查 `think → action[i+1]`，模型可用两轮 think 绕过：

```
[think] → [think] → [book_reservation("my_trip")]  ← v4 给 +0.01（越狱成功）
```

**v5 修复**：扩展到 `action[i+2]` 检查，2-step bypass 场景下第二个 think 不加分。

---

### P3：错误恢复分级（最重要修复之一）

**v4 问题**：错误后恢复不加区分，一律 +0.05。airline 中"同工具改参数"（换日期/支付方式）是最常见恢复，+0.05 过高，会强化随机试错。

**v5 修复**：三级区分

| 情况 | v4 | v5 | 理由 |
|------|-----|-----|------|
| 重复错误 | -0.04 | -0.04 | 不变 |
| 不同工具（策略调整） | +0.05 | +0.03 | 保守 |
| 同工具改参数（精细修正） | +0.05 | +0.02 | 精确奖励 |

**$305 事故案例**：
```
Turn 5: book_reservation(payment=[credit_card_305]) → Error: paid 305, should be 5
Turn 6: update_reservation_flights → Error: cannot change after booking
Turn 7: cancel_reservation → Success
Turn 8: book_reservation(payment=[certificate_250, credit_card_5]) → Success

→ v4: +0.05 +0.05 → 总和偏正，强化了试错
→ v5: +0.02 +0.02 → 更精确
```

---

### P8：长度惩罚阈值收紧

**v4 问题**：阈值 = 8，airline 优质轨迹集中在 4-6 步，几乎从不触发。

**v5 修复**：阈值 8→6，每步 -0.005（更温和）。

**工程逻辑**：超过 10 步的轨迹在 airline 上几乎全是失败案例（错误累积），早期惩罚让策略更早退出无效路径。

---

### B7：读工具多样性奖励阈值降低

**v4 问题**：要求 ≥3 种读工具，airline 只有 3 种，实际几乎不触发。

**v5 修复**：≥ 2 种即可触发（+0.01）。

---

### P5：两档无思考惩罚

**v4 问题**：只有 ≥3 步无 think 才惩罚，2 步无 think 无惩罚。

**v5 修复**：≥3 步 → -0.05；≥2 步 → -0.02（弱警告）。

---

## 完整评分维度

| 维度 | 标签 | v4 | v5 | 说明 |
|------|------|-----|-----|------|
| 工具错误 | P0 | -0.10 | -0.10 | v4 前缀；v5 语义 |
| 占位符参数 | P1 | -0.05/-0.03 | 不变 | schema 格式 |
| 冗余动作 | P2 | -0.03 | 不变 | 近 3 步相同 |
| 错误恢复 | P3 | +0.05/-0.04 | **+0.03/+0.02/-0.04** | v5 三级 |
| 无思考 | P5 | -0.05 | **-0.05/-0.02** | v5 两档 |
| 长度惩罚 | P8 | -0.01/步(>8) | **-0.005/步(>6)** | v5 收紧 |
| 读多样性 | B7 | +0.01(≥3) | +0.01(**≥2**) | v5 降低 |
| 软失败 | P0 | 无 | -0.02 | v5 新增 |

---

## 需要观测的指标

### 训练指标

| 指标 | 预期范围 | 含义 |
|------|---------|------|
| `train/group_saturation_rate` | < 10% | 全 0/全 1 group 比例 |
| `train/advantage_std_per_group` | > 0.3 | advantage 标准差 |
| `train/response_length_p50` | 100-400 tokens | 响应长度中位数 |
| `train/process_score_mean` | -0.1 ~ +0.2 | 过程分均值 |
| `train/process_outcome_conflict_rate` | < 20% | 反激励信号比例 |

### 评估指标

| 指标 | 含义 |
|------|------|
| `eval/pass_at_1_overall` | 整体 pass@1（核心） |
| `eval/pass_at_1_generalization` | 泛化 pass@1（排除训练集泄露） |
| `eval/error_rate` | 工具调用错误率，越低越好 |
| `eval/per_turn_p50` | 每轮 token 数 |

### PRM-Lite v5 诊断指标

| 指标 | 预期范围 | 含义 |
|------|---------|------|
| `prm/p0_soft_fail_detection_rate` | > 50% | 软失败检测覆盖率 |
| `prm/b4_b5_2step_bypass_rate` | < 5% | 2-step bypass 比例 |
| `prm/p3_recovery_bonus_avg` | +0.01 ~ +0.03 | 平均恢复奖励 |

---

## 快速开始

```bash
# PRM-Lite v5 训练
cd agentic-grpo-longhorizon/scripts/train/grpo
bash run_prm_lite.sh

# 验证 v5 评分逻辑（无需 verl）
python agentic-grpo-longhorizon/scripts/test/verify_prm_v5.py
```

---

## 核心结论

PRM-Lite v5 是零训练参数的过程奖励模型，通过 6 项工程优化（P0 软失败检测、P3 错误恢复分级等）系统性解决多轮 Agent 中的 GRPO 三大根因。Turn-PPO 在本项目中不使用，PRM-Lite v5 单独即可发挥作用。
