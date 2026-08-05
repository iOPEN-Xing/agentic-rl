# ABCD 用户模拟器适配：小规模 Pilot 审计

> 审计日期：2026-08-05  
> 当前可用版本：`abcd-usim-adapter-v0.3-pilot`  
> 结论：**14 条 pilot 可以作为独立的训练候选集；尚未授权全量生成，也没有混入现有主训练集。**

## 1. 这次适配解决的不是“格式转换”，而是训练语义对齐

ABCD 是电商领域的人类客服对话数据。它真正能补充当前 User Simulator 的能力是：逐步披露信息、针对 Agent 问题回答、一次回复多个被询问字段、在拒绝后继续推动 fallback，以及在长前缀中保持目标一致。它不能可靠提供当前运行时的终止边界，因此本 pilot 采用以下硬约束：

| ABCD 信息 | 当前 User Simulator SFT 中的处理 |
|---|---|
| `scenario.personal/order/product` | 转成客户已知的自然语言 system instruction |
| `flow/subflow` | 只用于经过人工审核的目标映射，不原样暴露给 Student |
| `agent` 自然语言 | 角色翻转为 SFT `user`，表示 Agent 对用户模拟器的输入 |
| `customer` 自然语言 | 作为 Teacher-only 语义锚点；DeepSeek 受约束改写为最后一个 SFT `assistant` target |
| `action` 事件和结果 | 完全过滤，不进入 scenario、Teacher context 或 Student messages |
| `end_conversation` / 末尾寒暄 | 不映射为 `###STOP###`，也不进入本 pilot |

训练时仍使用当前项目的 `last_assistant` loss mask：system、Agent 历史和历史 customer turn 都只作为条件，只有最后一条新 customer 回复计算 loss。

## 2. 为什么禁止 ABCD 生成 STOP

ABCD 的 `end_conversation` 描述源对话的流程结束，不等价于当前 τ-bench 运行时的“所有显式目标、条件 fallback 和用户要求的结果都已由 Agent 对用户可见地完成并传达”。如果直接映射，会把人类寒暄、客服主动结束和真实任务完成混为一谈，直接污染 premature/delayed STOP 边界。

因此本 pilot 的 14 条样本全部满足：

- `decision=continue`
- `is_over=false`
- `termination_reason=continue`
- target 不含 `###STOP###`
- `external_train_only=true`

ABCD 只补“如何继续”，现有 τ-bench 数据继续负责“何时结束”。

## 3. 三轮穿刺和修正

### v0.1：文本通过，但结构化状态语义错误

14 条生成文本都没有事实漂移，但 4 条把“客户这一句完整回答了 Agent 的问题”误写成 `communication_status=complete`。该字段实际上应描述 Agent 对客户整体业务目标的传达完成度。若不修正，未来把这些元数据用于 PRM、终止判定或数据筛选时会产生错误监督。

修正：明确 `communication_status` 的主体是 Agent 对整体业务目标的完成度，并把 `CONTINUE + complete` 设为硬失败。

### v0.2：状态主体正确，但背景动机被误当作业务目标

在 `timing_4` case 中，“想用优惠码给猫买帽子”只是背景动机，却被 Teacher 放进 `unresolved_goals`。这类目标膨胀会让模拟器在核心问题已解决后继续追问，形成 delayed STOP 倾向。

修正：scenario 明确分成 `Goal` 与 `Context (not a separate goal)`；为每个 pilot subflow 定义机器可检查的业务目标标签。

### v0.3：最终 pilot

Teacher 必须把每一个 `BUSINESS_GOAL_LABELS` 原样、且恰好一次地分配到 `resolved_goals` 或 `unresolved_goals`。两个集合必须互斥，并且并集必须等于预定义目标全集。该规则把“目标跟踪是否完整”从主观报告变成了可执行门禁。

## 4. 最终数据统计

| 指标 | v0.3 结果 |
|---|---:|
| 官方 sample 会话 | 3 |
| 审核后 case | 14 |
| CONTINUE / STOP | 14 / 0 |
| DeepSeek 首次生成通过 | 14 / 14 |
| 生成重试 | 0 |
| 事实/标识符漂移 | 0 |
| action/tool 文本泄露 | 0 |
| 运行时 system prompt 精确匹配 | 14 / 14 |
| Student prefix 精确匹配 | 14 / 14 |
| 业务目标完整互斥分区 | 14 / 14 |
| `communication_status=complete` | 0 |
| 与当前主训练集 case_id 重叠 | 0 |
| 与源 customer 文本完全相同 | 4 / 14 |
| DeepSeek v0.3 tokens | 14,575 prompt + 1,973 completion = 16,548 |

源文件 SHA-256：`151e0c487493ab376bb5115538f3bfd6d2f460c94f9daa5cdf04e55bccdf4808`。  
v0.3 生成文件 SHA-256：`c97f9ef3c8a390a20836c071a90ff3548f69b4a321f0f3d9122b81cd2102445b`。  
TRL 训练文件 SHA-256：`9a9c9c5c77f3bbc78c0ee86dd09d4998a0e5abcc74c288bcfbd2cd6d68193b12`。

## 5. 逐条人工语义审核

| Case | 训练行为 | 人工结论 |
|---|---|---|
| `c03592-u002` | 首轮只提出退货，不一次性倾倒身份信息 | 通过 |
| `c03592-u004` | Agent 询问姓名后仅提供姓名 | 通过 |
| `c03592-u007` | Agent 询问原因后说明尺码错误 | 通过 |
| `c03592-u009` | 一次回答 Agent 同时询问的 username、email、order ID | 通过 |
| `c03592-u014` | 只回答会员等级 | 通过 |
| `c03592-u016` | 回答是否在 90 天内购买，并保持原始时间语义 | 通过 |
| `c03592-u018` | 正常退货被拒后继续推动，而不是提前结束 | 通过 |
| `c03592-u021` | Agent 提供经理 escalation 后，按要求提供电话 | 通过 |
| `c03695-u002` | 提出优惠码有效期问题；给猫买帽子仅作为背景 | 通过 |
| `c03695-u010` | 澄清未尝试使用，只想知道有效期 | 通过 |
| `c09489-u001` | 首轮提出查询退款状态 | 通过 |
| `c09489-u003` | Agent 询问姓名或账户信息后提供姓名和 username | 通过 |
| `c09489-u008` | 同时提供 order ID 与 email，无新增标识符 | 通过 |
| `c09489-u016` | Agent 已传达退款处理中，继续追问预计完成时间 | 通过 |

代表性长前缀 target：

```text
Agent: It is currently in progress and the payment method with which it is being
       processed is online towards your credit card.

Human source: how much long till it is refunded
DeepSeek v0.3 target: How much longer until it's refunded?

resolved_goals:
  - learn the refund status
unresolved_goals:
  - learn the approximate refund completion time
decision: continue
```

这个 case 体现了本项目真正需要的状态语义：Agent 已完成“状态查询”子目标，但还没有回答“预计多久完成”，所以用户应继续追问，不能 STOP，也不能重复询问退款当前状态。

## 6. 产物与复现

本地训练候选产物位于：

```text
outputs/user_simulator_data/abcd_adapter/
├── abcd_sample.official.json
└── pilot/
    ├── source_cases.jsonl
    ├── source_manifest.json
    ├── v0.1.jsonl                 # 失败分析保留，不用于训练
    ├── v0.2.jsonl                 # 失败分析保留，不用于训练
    ├── v0.3.jsonl                 # 当前审核通过的丰富记录
    ├── v0.3.report.json
    └── sft/
        ├── train.jsonl            # 14 条 rich SFT
        ├── train_trl.jsonl        # 14 条仅 messages 的训练输入
        └── manifest.json
```

复现命令：

```bash
python3 scripts/train/user_simulator/build_abcd_pilot.py

# 在当前 shell 安全设置 DEEPSEEK_API_KEY；不要写入脚本或 Git。
python3 scripts/train/user_simulator/generate_abcd_pilot.py --workers 2

python3 scripts/train/user_simulator/export_sft.py \
  --input outputs/user_simulator_data/abcd_adapter/pilot/v0.3.jsonl \
  --cases outputs/user_simulator_data/abcd_adapter/pilot/source_cases.jsonl \
  --output outputs/user_simulator_data/abcd_adapter/pilot/sft/train.jsonl \
  --trl-output outputs/user_simulator_data/abcd_adapter/pilot/sft/train_trl.jsonl \
  --manifest-output outputs/user_simulator_data/abcd_adapter/pilot/sft/manifest.json \
  --expected-prompt-version abcd-usim-adapter-v0.3-pilot \
  --stop-repeat 1
```

## 7. 是否应该扩量

当前结论是“pilot 数据本身满意，可以进入下一阶段设计”，不是“已经证明 ABCD 应大规模混入训练”。14 条只证明转换语义、DeepSeek 约束和质量门禁可用，不能证明跨 55 个 intent 的覆盖，也不能证明对航空领域指标有净收益。

在用户确认前，`full_scale_generation_allowed` 保持 `false`。后续若扩量，至少要先满足：

1. 下载并锁定官方 ABCD v1.1 train split，不使用 dev/test。
2. 为进入候选池的每个 subflow 人工审核 `Goal / Context / fallback` 映射；未知 subflow 继续 fail closed。
3. 排除寒暄、纯情绪、小聊、action 依赖回复和结束轮次。
4. 外部数据继续保持 0 个 STOP，并与当前 τ-bench eval/holdout 完全隔离。
5. 第一轮只做单一来源 A/B，建议将 ABCD 控制在训练 batch 的约 5%，监控航空领域事实漂移、过度配合、premature/delayed STOP 和主任务成功率。
6. 不能把 ABCD 静态 human-prefix 数据当成 Student-prefix rollout 的替代品；后者仍是解决 exposure bias 的优先数据来源。

官方来源：[ABCD GitHub](https://github.com/asappresearch/abcd)、[NAACL 2021 论文](https://aclanthology.org/2021.naacl-main.239/)。
