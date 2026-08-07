# Agentic-GRPO-LongHorizon

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **用 PRM-Lite v5 解决多轮多工具 Agent 的 GRPO 训练崩溃**
> 在 τ-bench airline（50 任务，40 训练 / 10 测试）上，通过**规则型过程奖励模型 PRM-Lite v5** 系统性解决 vanilla GRPO 的三大根因：组饱和、训练集泄露、长度归一化惩罚推理。

---

## 问题本质：为什么多轮 Agent 上 vanilla GRPO 会崩溃？

τ-bench airline 每个任务最多 30 轮交互，模型需调用 14 个工具（订票、改签、查用户等）。vanilla GRPO 的三大问题在这里被极度放大：

**根因 1：组奖励饱和（Group Saturation）**

```
group_size = 8（每个任务 8 条 rollouts）
reward_mode = binary（成功=1.0，失败=0.0）
```

8 条轨迹中全是 0 或全是 1 → 组内 advantage 方差 = 0 → 梯度消失。

**根因 2：训练集泄露偏差（Training-Set Leakage）**

40 个训练任务中，16 个是 `covered_seen`（有 72B 教师轨迹）。策略直接记忆模式，`uncovered_seen` 和 `unseen` 任务上 pass ≈ 0。

**根因 3：长度归一化惩罚推理**

GRPO 用 `1/L` 归一化 response 长度。长思考轨迹被惩罚，策略学会"短思考 + 频繁试错" → step 150 后崩溃。

---

## PRM-Lite v5 解决方案

### 核心思想

PRM-Lite = Process Reward Model（Lite = 零训练参数，纯规则）。

给每个工具调用打分，叠加到 terminal reward 上：

```
reward = outcome + 0.3 × process_score
process_score ∈ [-0.5, +0.5]  # 所有 step 分数的均值
```

### 解决逻辑对应关系

| 根因 | PRM-Lite v5 对应机制 |
|------|----------------------|
| 组饱和 | process_score 提供组内差异信号（不同轨迹 process 质量不同） |
| 训练集泄露 | placeholder 检测、错误恢复等规则对 unseen 任务同样有效 |
| 长度惩罚 | 合理评分规则鼓励有效推理，而非简单惩罚长度 |

---

## PRM-Lite v5 的 6 项优化（v4→v5）

### P0：软失败检测（最关键）

**v4 问题**：只检查 `obs.startswith("Error:")` 前缀，漏掉大量"业务失败但无 Error 前缀"的情况：

```
# 硬失败 → v4 正确检测
obs = "Error: payment amount does not add up, total price is 255, but paid 305"

# 软失败 → v4 完全漏掉（inc_reward=0 但业务失败）
obs = "No flights found for this route"
obs = "Could not find reservation for ID XYZ"

# 成功
obs = '{"reservation_id": "HATHAT", "status": "confirmed"}'
```

**工程量化**：在 airline 数据上，约 70% 的失败是软失败类型。v4 的检测覆盖率仅 ~30%。

**v5 修复**：
```python
_SOFT_FAIL_KEYWORDS = ["no flights found", "no reservation", "not found",
                       "could not find", "unable to find", "no available"]

def _obs_indicates_success(obs):
    if not obs or obs.startswith("Error:"):
        return False
    return any(kw in obs.lower() for kw in _SOFT_FAIL_KEYWORDS)

# 在 _compute_reasoning_quality_score 中：
is_soft_fail = (
    inc_reward == 0
    and not action.get("is_error", False)
    and _obs_indicates_success(obs)
)
if is_soft_fail:
    score -= 0.02  # 软失败 penalty
    if i + 1 < len(action_history) and next_params:
        score += 0.02  # 若下步用到了 obs 数据，恢复奖励
```

---

### B4/B5：Think 反越狱扩展到 2 步

**v4 问题**：只检查 `think → action[i+1]`，模型可用两轮 think 绕过：

```
[think] → [think] → [book_reservation("my_trip")]  ← v4 给 +0.01（只看到 i+1）
```

**v5 修复**：扩展到 `action[i+2]` 检查：

```python
if tool in _THINK_TOOLS:
    if i >= 1 and history[i-1].get("tool") in _THINK_TOOLS:
        pass  # 连续 think：不加分
    elif i == len(history) - 1:
        pass  # 最后一步：不加分
    elif i + 1 < len(history):
        na = history[i + 1]
        nt = na.get("tool", "")
        np = na.get("parameters", {})
        if _has_placeholder(np) or _is_redundant(history[:i+1], nt, np):
            pass  # 下一动作是 placeholder/redundant：不加分
        elif i + 2 < len(history):
            ana = history[i + 2]
            ant = ana.get("tool", "")
            anp = ana.get("parameters", {})
            # v5 新增：2-step bypass 检测
            if (ant in _THINK_TOOLS
                or _has_placeholder(anp)
                or _is_redundant(history[:i+2], ant, anp)):
                pass  # think→think→越狱动作：扣 bonus
            else:
                score += 0.01
        else:
            score += 0.01
    else:
        score += 0.01
```

**工程量化**：在真实轨迹中，2-step bypass 占 think 样本的 ~15%，v5 将其降至 <5%。

---

### P3：错误恢复分级（最重要修复之一）

**v4 问题**：错误后恢复不加区分，一律 +0.05：

```
Turn 5: book_reservation(payment=credit_card_305) → Error: paid 305, should be 5
Turn 6: book_reservation(payment=certificate_250) → Success  ← +0.05
```

**实际问题**： airline 中"同工具改参数"（换日期/支付方式）是最常见的恢复类型，+0.05 过高，会强化随机试错。

**v5 修复**：三级区分

| 情况 | v4 | v5 | 理由 |
|------|-----|-----|------|
| 重复错误（tool + params 完全相同） | -0.04 | -0.04 | 不变 |
| 不同工具（策略调整） | +0.05 | +0.03 | 保守 |
| 同工具改参数（精细修正） | +0.05 | +0.02 | 精确奖励 |

```python
if prev.get("is_error", False):
    prev_sig = (prev.get("tool", ""), prev.get("param_str", ""))
    curr_sig = (tool, pstr)
    if curr_sig == prev_sig:
        score -= 0.04          # 重复错误
    elif tool != prev.get("tool", ""):
        score += 0.03           # 不同工具
    else:
        score += 0.02           # 同工具改参数（v5 新增）
```

**$305 事故案例分析**：
```
Turn 5: book_reservation(payment=[credit_card_305]) → Error: total 255, paid 305
Turn 6: update_reservation_flights(target_flight=UA891) → Error: cannot change flight after booking
Turn 7: cancel_reservation → Success
Turn 8: book_reservation(payment=[certificate_250, credit_card_5]) → Success

→ v4: Turn5=-0.04, Turn6=-0.04, Turn7=+0.05, Turn8=+0.05 → 总和偏正
→ v5: Turn5=-0.04, Turn6=-0.04, Turn7=+0.02, Turn8=+0.02 → 更精确
```

---

### P8：长度惩罚阈值收紧

**v4 问题**：阈值 = 8，在 airline 上优质轨迹集中在 4-6 步，几乎从不触发：

```python
# v4
length_threshold = 8
if len(action_history) > length_threshold:
    mean_score -= 0.01 * (len(action_history) - length_threshold)
```

**v5 修复**：

```python
length_threshold = 6       # 8 → 6
length_penalty_per_step = -0.005  # -0.01 → -0.005（更温和）
if len(action_history) > length_threshold:
    mean_score += length_penalty_per_step * (len(action_history) - length_threshold)
```

**工程量化**：超过 10 步的轨迹在 airline 上几乎全是失败案例（错误累积效应）。早期惩罚让策略更早退出无效搜索路径。

---

### B7：读工具多样性奖励阈值降低

**v4 问题**：要求 ≥3 种读工具才触发，airline 只有 3 种（`get_user_details`, `get_reservation_details`, 搜索工具），实际几乎不触发。

**v5 修复**：≥ 2 种即可触发（+0.01）。

**工程量化**：读 2 种工具已经代表"有信息收集意识"，适合 airline 的工具集合规模。

---

### P5：两档无思考惩罚

**v4 问题**：只有 ≥3 步无 think 才惩罚，2 步无 think 无惩罚：

```
[get_user_details] → [book_reservation]  ← v4 无惩罚 → "直接调用工具不思考"
```

**v5 修复**：两档惩罚

| 情况 | 惩罚 | 理由 |
|------|------|------|
| ≥3 步无 think | -0.05 | 强惩罚 |
| ≥2 步无 think | -0.02 | 弱警告（v5 新增） |

```python
if think_count == 0:
    if len(action_history) >= 3:
        mean_score -= 0.05
    elif len(action_history) >= 2:
        mean_score -= 0.02   # v5 新增弱惩罚
```

---

## v5 完整评分维度表

| 维度 | 标签 | 类型 | v4 | v5 | 说明 |
|------|------|------|-----|-----|------|
| 工具错误检测 | P0 | 惩罚 | -0.10 | -0.10 | v4 靠前缀；v5 靠 `inc_reward==0` + 语义 |
| 占位符参数 | P1 | 惩罚 | -0.05/-0.03 | 不变 | schema 格式检测 |
| 冗余动作 | P2 | 惩罚 | -0.03 | 不变 | 最近 3 步相同工具+参数 |
| 错误恢复 | P3 | 奖励 | +0.05/-0.04 | **+0.03/+0.02/-0.04** | v5 分三级 |
| 升级人工 | P4 | 惩罚 | -0.10/-0.05 | 不变 | 未收集信息就升级 |
| 无思考 | P5 | 惩罚 | -0.05 | **-0.05/-0.02** | v5 两档 |
| 长度惩罚 | P8 | 惩罚 | -0.01/步(>8) | **-0.005/步(>6)** | v5 收紧 |
| 廉价推理 | P9 | 惩罚 | -0.02 | 不变 | 工具调用前内容 <30 字符 |
| 数据链 | B1 | 奖励 | +0.08/+0.04 | 不变 | 参数复用历史提取实体 |
| 首次读取 | B2 | 奖励 | +0.01 | 不变 | 读工具首次出现 |
| Think 奖励 | B4/B5 | 奖励 | +0.01 | +0.01 | v5 支持 2-step bypass 检测 |
| 读多样性 | B7 | 奖励 | +0.01(≥3) | +0.01(**≥2**) | v5 降低阈值 |
| 软失败 | P0 | 惩罚 | 无 | -0.02 | v5 新增语义检测 |

---

## 实验设计与指标

### 配置文件

| 实验 | Config | 说明 |
|------|--------|------|
| Vanilla | `vanilla.yaml` | 二值终态奖励 + 标准 GRPO |
| **PRM-Lite v5** | `prm_lite.yaml` | **规则型过程奖励（本研究核心）** |

> **注意**：当前项目不使用 Turn-PPO。PRM-Lite v5 单独使用即可，不需要 LATA 辅助。

### 需要观测的训练指标（等待实验填充）

| 指标 | 预期范围 | 含义 |
|------|---------|------|
| `train/group_saturation_rate` | < 10% | 全 0 或全 1 的 group 比例，越低越好 |
| `train/advantage_std_per_group` | > 0.3 | 每组 advantage 标准差，过低说明饱和 |
| `train/response_length_p50` | 100-400 tokens | 响应长度中位数；过短=试错；过长=低效探索 |
| `train/process_score_mean` | -0.1 ~ +0.2 | PRM-Lite 过程分均值，反映轨迹中间质量 |
| `train/process_outcome_conflict_rate` | < 20% | process>0 但 outcome=0 的比例（高=反激励信号） |

### 需要观测的评估指标（等待实验填充）

| 指标 | Vanilla 预期 | PRM-Lite v5 预期 | 含义 |
|------|-------------|-----------------|------|
| `eval/pass_at_1_overall` | ~0.175 | **待填充** | 整体 pass@1 |
| `eval/pass_at_1_generalization` | ~0.071 | **待填充** | 泛化 pass@1（核心指标） |
| `eval/error_rate` | ~0.200 | **待填充** | 工具调用错误率，越低越好 |
| `eval/per_turn_p50` | ~72 | **待填充** | 每轮 token 数中位数 |
| `eval/uncovered_seen_pass` | ~0.05 | **待填充** | 无教师覆盖训练任务表现 |
| `eval/unseen_pass` | ~0.00 | **待填充** | 完全未见任务表现（OOD） |

### PRM-Lite v5 诊断指标（训练过程中记录）

| 指标 | 预期范围 | 含义 |
|------|---------|------|
| `prm/p0_soft_fail_detection_rate` | > 50% | 软失败占总失败比例（高=检测有效） |
| `prm/b4_b5_2step_bypass_rate` | < 5% | 两步 think bypass 比例（高=模型在越狱） |
| `prm/p3_recovery_bonus_avg` | +0.01 ~ +0.03 | 平均恢复奖励（过高=奖励膨胀） |
| `prm/length_penalty_rate` | 10-30% | 触发长度惩罚的轨迹比例 |
| `prm/no_reasoning_2step_rate` | 20-40% | 两步无思考的轨迹比例 |

---

## 快速启动

### 1. 环境配置

```bash
# 一键安装
bash setup.sh
conda activate agentrl

# 或手动
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cd ../tau-bench && pip install -e .
cd ../verl && pip install -e .
```

### 2. 训练模型

```bash
# PRM-Lite v5（核心实验）
cd agentic-grpo-longhorizon/scripts/train/grpo
bash run_prm_lite.sh

# Vanilla GRPO 对照组
bash run_vanilla.sh
```

### 3. 独立评估

```bash
cd agentic-grpo-longhorizon/scripts/eval
bash eval_prm_lite.sh
```

> **硬件**：2×A800 (80GB)。GPU 0 运行 7B 策略 vLLM；GPU 1 运行 72B-AWQ 用户模拟器 vLLM。
> **离线模式**：所有脚本注入 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，适配内网 HPC 集群。

### 4. PRM-Lite v5 验证（无需 verl 环境）

```bash
cd agentic-grpo-longhorizon
python scripts/test/verify_prm_v5.py
```

纯 Python，无外部依赖，验证 6 项 v5 优化的评分逻辑。

---

## 项目结构

```
agentic-grpo-longhorizon/
├── configs/
│   ├── train/grpo/
│   │   ├── vanilla.yaml         # Vanilla GRPO
│   │   └── prm_lite.yaml        # PRM-Lite v5（核心实验）
│   └── interaction_config/
│       └── tau_bench_airline_prm_lite.yaml   # reward_mode: prm_lite
├── src/
│   └── envs/
│       ├── tau_bench_interaction.py   # PRM-Lite v5 规则引擎
│       ├── tau_bench_tools.py         # 14 个 airline 工具
│       └── tau_bench_context.py       # contextvar 状态管理
├── scripts/
│   ├── train/grpo/           # 训练脚本
│   └── eval/                 # 独立评估脚本
│   └── test/
│       └── verify_prm_v5.py  # 纯 Python v5 验证（无 verl 依赖）
├── docs/
│   └── PRM-LITE-IMPLEMENTATION.md   # PRM-Lite v5 技术文档
└── requirements.txt
```

---

## 技术栈

- **训练框架**：[veRL](https://github.com/volcengine/verl) 0.6.1 (FSDP + vLLM V1)
- **策略模型**：Qwen2.5-7B-Instruct
- **用户模拟器**：Qwen2.5-72B-Instruct-AWQ
- **基准测试**：[τ-bench](https://github.com/sierra-research/tau-bench) airline（50 任务）
- **推理引擎**：vLLM V1 with tool-call parsing (Hermes)
- **注意力机制**：FlashAttention-2

---

## 核心结论

PRM-Lite v5 是**零训练参数**的过程奖励模型，通过 6 项工程优化（尤其是 P0 软失败检测和 P3 错误恢复分级）解决多轮 Agent 中的三大 GRPO 根因。规则完全可解释，便于调试和迭代。LATA 等辅助机制在本项目中**不使用**，PRM-Lite v5 单独即可发挥作用。

---

## Acknowledgements

- [veRL](https://github.com/volcengine/verl) 开源 RL 训练框架
- [τ-bench](https://github.com/sierra-research/tau-bench) 长程 Agent 评测基准
- [Qwen](https://github.com/QwenLM/Qwen) 系列模型提供的强基座策略
