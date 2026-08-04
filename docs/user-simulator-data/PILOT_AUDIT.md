# DeepSeek V4 Flash User Simulator Pilot 审计

> 状态：通过。本文只记录 16 个 curated case × 3 次采样的真实 pilot；后续 1,513-case full generation 已完成，结果见 [`FULL_GENERATION_AUDIT.md`](FULL_GENERATION_AUDIT.md)，Qwen3-14B 微调尚未启动。

## 1. 最终结论

最终采用以下 teacher 配置：

```json
{
  "model": "deepseek-v4-flash",
  "thinking": false,
  "temperature": 0.3,
  "top_p": 0.9,
  "max_tokens": 1600,
  "prompt_version": "tau-airline-usim-teacher-v1.4"
}
```

`v1.7` 的 48 次真实请求结果：

| 指标 | 结果 |
|---|---:|
| accepted | 48 / 48 |
| curated STOP/CONTINUE match | 48 / 48 |
| STOP case | 12 / 12 精确输出 `###STOP###` |
| CONTINUE case | 36 / 36 继续完成真实未决目标 |
| JSON parse / non-empty | 48 / 48 |
| privileged entity leak | 0 |
| case-specific semantic violation | 0 |
| contract issue | 0 |
| `quality_gate_passed` | `true` |

真实报告保存在本地 gitignored 路径：

```text
agentic-grpo-longhorizon/outputs/user_simulator_data/pilot/v1.7.report.json
```

API key、provider reasoning、Authorization header 和原始私有推理均不写入结果。

## 2. 为什么关闭 thinking

第一版使用 high thinking 和 `max_tokens=1600`。30 个可解析样本的 completion token 中位数为 881.5、P90 为 1,408、最大为 1,574，已经逼近 1,600 上限。最终造成：

- 12 个空 `content`；
- 6 个截断或不完整 JSON；
- accepted 仅 30 / 48。

把预算提高到 4,096 可以消除截断，但这个任务的主要产物是“终止边界分类 + 一句短用户回复”，并不需要长 reasoning。关闭 thinking 后，每条 completion 的中位数约 145 token、最大 184 token，48 条都能稳定输出完整 JSON，且不保存不可审计的 chain-of-thought。

## 3. 为什么 temperature 从 0.8 降到 0.3

`temperature=0.8` 的语言变化更多，但会让任务语义漂移：

- reactive 用户在首轮提前披露时间、舱位、转机和价格偏好；
- 已完整完成 fallback/cancellation 后仍输出感谢或再次确认，而不是 STOP；
- compound task 已给出结果后仍重复询问。

当前数据的目标不是生成客服文案，而是训练一个稳定的 RL 环境。对 User Simulator 来说，STOP precision、STOP recall、目标坚持和不泄漏必须优先于表面措辞变化。因此最终使用 `temperature=0.3, top_p=0.9`，多样性主要来自 1,513 个 runtime-reachable seen-task 对话状态、不同 persona、不同未完成子目标和自然历史，而不是让同一个决策边界高温漂移。

## 4. 多样性不是“每句话必须不同”

最终 pilot 的 36 条 CONTINUE 回复中：

- 25 条是不同的规范化文本；
- exact duplicate rate 为 30.56%；
- 排除必须精确回答的 user ID 后，11 个可变 case 组中 9 个产生了多种表达。

重复并不全部是坏事：

- STOP 必须固定为 `###STOP###`，不能追求改写；
- 被问 user ID 时，`My user ID is mia_li_3668.` 是最短且最可靠的回答；
- “所有修改已完成但总节省未告知”时，简洁追问 total savings 比增加风格噪声更重要；
- confirmation 需要明确肯定，不能为多样性生成含糊表达。

允许变化的轴是 contractions、礼貌程度、句式和符合 persona 的简洁度；禁止变化的轴是实体、金额、确认状态、未决目标、fallback 和 STOP 边界。

## 5. Pilot 如何覆盖当前两个核心问题

| 边界 | Case | 期望 | 训练含义 |
|---|---|---|---|
| 初始 reactive disclosure | task 0 | CONTINUE | 只说订票、航线和日期，不一次倾倒所有槽位 |
| 必要 ID | task 0 | CONTINUE | 被问时提供 scenario 中真实 user ID |
| profile-only DOB | task 0 | CONTINUE | 要求 Agent 查 profile，不虚构或泄漏 DOB |
| write confirmation | task 0 | CONTINUE | 明确确认，不把待执行动作误判完成 |
| booking 完整告知 | task 0 | STOP | 不增加礼貌尾轮 |
| 五个 reservation 只完成三个 | task 2 | CONTINUE | 坚持剩余两项和 total savings |
| DB 写操作完成、output 缺失 | task 2 | CONTINUE | 不用 STOP 掩盖 Agent 漏报 |
| state + output 都完整 | task 2 | STOP | 正确结束 |
| 第三个 Agent 消息触发子目标 | task 34 | CONTINUE | 询问其他 upcoming flights 和总价 |
| compound goal 全部完整 | task 34 | STOP | 取消、航班明细与 $1,016 都告知后结束 |
| 第一次 malformed total | task 34 | CONTINUE | 要求可读的精确数字 |
| 多次 malformed total | task 34 | CONTINUE | persistent persona 不被错误训练成放弃 |
| basic-economy fallback | task 1 | CONTINUE | 触发 insurance cancellation |
| fallback 已完成 | task 1 | STOP | 不再次确认已经确认的退款 |
| Agent 假称三项都完成 | task 4 | CONTINUE | 指出 baggage 仍未完成 |
| reservation ID 不记得 | task 1 | CONTINUE | 请求按账户/姓名查找，不泄漏 gold ID |

这组 case 明确区分：

```text
工具/DB 已成功，但用户要求的结果未告知  => CONTINUE
Agent 回复 malformed 或重复              => CONTINUE，暴露 policy failure
所有显式/条件目标都已完成且已告知          => ###STOP###
```

因此它不会通过“让 User 更早 STOP”来虚增主 Agent 的终局 reward。

## 6. task 34 的反向数据修复

Pilot 迭代发现，早期 compound-complete case 只列了 `7WPL39`、`3EMQJ6`，却声称总价为 `$1,016`。真实 τ-bench 状态还有 `A90KR2`：

```text
7WPL39 = $402
3EMQJ6 = $306
A90KR2 = $308
total  = $1,016
```

`402 + 306 + 308 = 1,016`。当前 case 已按仓库内真实 task 34 历史轨迹修正，并明确告诉用户这是完整的其他 upcoming flight 集合。修复前 teacher 继续追问是合理行为，不能将它强行标为 STOP。这说明 teacher pilot 不只是测试模型，也能反向发现人工合成 case 的 evaluator/事实缺口。

## 7. 新增的可执行语义门禁

仅检查 `expected_decision` 不足以保证训练数据质量。当前 curated case 支持：

```json
{
  "response_constraints": {
    "must_include_any_of_each": [
      ["book", "flight"],
      ["new york", "nyc", "jfk"],
      ["seattle", "sea"]
    ],
    "must_not_include": [
      "economy",
      "insurance",
      "certificate",
      "7447"
    ],
    "max_chars": 160
  }
}
```

每个内层数组允许多个自然同义表达，不强迫一个固定模板；但关键语义和提前披露禁区是硬门禁。当前覆盖 progressive disclosure、confirmation、fallback、missing output、malformed recovery、baggage omission 和 unknown ID。

## 8. 版本迭代记录

| 版本 | 关键变化 | 真实结果 | 判断 |
|---|---|---:|---|
| v1.0 | high thinking，1,600 output token | 30 / 48 | reasoning 挤占 JSON，拒绝 |
| v1.1 | high thinking，4,096 token，reactive prompt | 中途停止；31 条已返回均 accepted | 稳定但成本/延迟无必要 |
| v1.2 | thinking off，T=0.8 | 45 / 48 | 有 delayed STOP，拒绝 |
| v1.3 | 补无历史 evidence 与 courtesy STOP 规则 | 48 / 48 | 硬合同通过，但人工发现首轮过披露 |
| v1.4–v1.6 | 加 case 语义门禁，修 task 34 完整事实 | 暴露高温语义漂移和门禁同义词误杀 | 继续校准 |
| v1.7 | thinking off，T=0.3，top-p=0.9 | 48 / 48 | 最终 pilot 通过 |

## 9. 当前放行边界

在当时，这一阶段只证明：16 类关键边界在 3 次随机采样下稳定，prompt、合同和 non-thinking 调用配置可进入全量阶段。它没有单独证明：

- 1,513 个 runtime-reachable 历史状态全部能自动通过；
- 微调后的 Qwen3-14B 在线 STOP latency 一定改善；
- 主 Agent terminal success 一定提升；
- unseen 10-task benchmark 的泛化已经成立。

后续 full generation 已按这些条件执行：1,513/1,513 自动门禁通过、0 privileged leak，并完成 70 个决策分歧的逐条语义审计。该结果不反向改变本页的 pilot 统计；微调后的同 simulator / cross-simulator 在线评估仍待执行。
