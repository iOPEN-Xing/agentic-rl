# 开源用户模拟器数据：当前项目适配审计

> 审计日期：2026-08-05
>
> 适配目标：当前项目的 τ-bench airline User Simulator，而不是通用聊天模型
>
> 当前结论：**只保留 ABCD 的 14 条隔离 pilot；不新增其他数据，不启动全量生成。**

## 1. 先定义“有用”，避免把多轮对话误当用户模拟器数据

当前 User Simulator 的训练目标不是简单预测下一句，而是：给定完整场景和 Agent 可见回复，在不读取工具隐状态的前提下，持续维护客户的显式目标、条件 fallback、已披露事实和终止边界。只有同时满足下列条件的数据，才可能带来净收益：

1. 有明确的用户目标、约束或结构化状态，可证明用户回复与目标一致；
2. 对话中能分离 Agent 自然语言、Customer 自然语言和不可见的工具/action 状态；
3. 可以只使用 train split，并能与当前 τ-bench eval/holdout 隔离；
4. 许可证允许当前用途；
5. 能映射到当前运行时的 `system + Agent/User history -> next customer response`；
6. 外部语料不能擅自提供 `###STOP###`，除非其结束条件与当前运行时逐项等价；
7. 数据带来的目标保持、渐进披露或恢复行为价值，必须大于领域漂移、模板化和错误终止监督的风险。

因此，“规模大”“对话轮数多”或“名字里含 user simulation”都不是准入理由。

## 2. 严格筛选结果

| 数据集 | 可迁移价值 | 关键风险 | 当前决定 |
|---|---|---|---|
| **当前 τ-bench 历史轨迹 / Student-prefix rollout** | 与运行时、场景、工具环境和 STOP 协议完全一致 | 历史标签噪声、teacher/student exposure gap | **主数据，优先级最高** |
| **ABCD v1.1** | 10K+ 人类客服对话、55 intents；适合渐进披露、多字段回答、拒绝后继续推动 | 电商跨域；action 不可见；`end_conversation` 不等于当前 STOP | **仅 14 条 CONTINUE pilot 已准入，尚未批准扩量** |
| **AirDialogue** | 航空领域、人类 customer-agent 对话、显式 intent/expected action，领域最接近 | 任务主要是旧式 booking/change/cancel；Agent 的业务状态机与 τ-bench 不同；数据卡为 CC-BY-NC-4.0 | **高优先级候选，但先做许可证和 20–30 条静态 pilot 审计** |
| **Schema-Guided Dialogue (SGD/SGD-X)** | 20K+ 多域任务对话，有 intents、slots、service calls；官方明确列出 user simulation learning 用途 | outline 由 simulator 生成再由众包改写；跨域；服务/API 与当前环境不一致；CC-BY-SA-4.0 | **仅可作为低比例语言/槽位恢复候选，不提供 STOP** |
| **MultiWOZ** | 10K 人类多域任务对话，有 goal 和 belief state | 官方说明存在未完成目标和错误 belief state；用户侧 dialogue acts 并非全部人工标注；无当前工具状态语义 | **暂不加入** |
| **Taskmaster** | 55K+ 口语/书面任务对话，能增加表达多样性 | 包含 self-dialog；目标/状态监督弱于前三者；仓库顶层没有清晰数据许可证声明 | **许可证和语义门未通过，暂不加入** |
| **Humanual-Chat / RealUserSim** | 真实 LLM 用户的追问、澄清和风格画像 | 通用开放域、没有可执行业务目标或可靠完成状态；可能放大闲聊和 goal drift | **不用于当前主 SFT；最多未来做独立风格评测** |
| **SpokenTOD / SpokenUS** | 有打断、口误、情绪和跨轮 slot 等语音行为 | 当前项目是文本 User Simulator；其收益主要针对语音链路 | **当前不加入** |

## 3. 为什么先做 ABCD，而不是直接灌入 AirDialogue

AirDialogue 的领域相似度最高，但“航空”相同不等于状态语义相同：

- AirDialogue 的结构化 `action` / `expected_action` 是源环境的最终业务状态，不能作为当前 Student 可见上下文；
- 源数据中的礼貌结束、人工对话结束或预订 action 正确，不等于当前场景中每个显式目标、fallback 和 requested output 都已经由 Agent 对用户传达；
- AirDialogue 的大规模样本集中在 book/change/cancel，当前 τ-bench airline 还包含 cabin、baggage、insurance、certificate、refund、乘客修改和组合任务；未经重建会产生“同领域但错误策略”的负迁移；
- 官方 GitHub 代码仓库是 Apache-2.0，但官方 Hugging Face 数据卡将数据标为 CC-BY-NC-4.0。代码许可证不能替代数据许可证，商业用途必须先单独确认。

ABCD 的优势不是领域，而是行为：人工客户会逐步披露身份信息、针对 Agent 的具体问题作答、在拒绝后继续推进 fallback。该数据仓库本身使用 MIT，且当前适配器已经把 `action`、结束轮次和 STOP 全部隔离。因此它更适合先验证“外部人类对话能否提升用户行为”，但仍只能从小比例 pilot 开始。

## 4. ABCD pilot 的准入边界

当前已经审核的 ABCD v0.3 只有 14 条：

- 全部是 `CONTINUE`，0 条外部 STOP；
- action/tool 文本完全不可见；
- 场景目标按具体 conversation 绑定，未知会话 fail closed；
- 每条样本的业务目标必须在 resolved/unresolved 中做完整且互斥的分区；
- DeepSeek 只对 human-authored customer target 做受约束改写，不负责创造任务、事实或终止标签；
- 数据保持为独立候选集，没有混入主训练集；
- 当前 `full_scale_generation_allowed=false`。

全量 ABCD train 的静态结构过滤后存在 44,110 个 CONTINUE 候选上界，但其中仍包含小聊、无信息确认和电商专有流程。这个数字不是可训练样本数，更不是应当调用 Teacher 的数量。

详细实现和审计见 [ABCD_PILOT_AUDIT.md](./ABCD_PILOT_AUDIT.md)。

## 5. 下一步的最小验证顺序

### 阶段 A：先让用户确认现有 14 条 ABCD pilot

不改训练配置，不合并主数据。人工确认以下四类 case：渐进披露、多字段回答、拒绝后 fallback、部分目标已完成但仍需继续。

### 阶段 B：ABCD 受控扩展，而非全量

用户确认后，按 55 个 canonical intents 各抽 2 个 train 会话，每个会话最多 2 个目标轮次，理论上限 220 条。每个会话单独审核 Goal / Context / fallback，不能复用 subflow 全局模板。

### 阶段 C：单独做 AirDialogue 静态 pilot

在确认数据许可证适用后，先抽 20–30 条 train 样本，只验证：

1. intent 是否能无损改写为当前客户已知场景；
2. customer 回复是否只依赖场景与可见 Agent 文本；
3. 是否能剔除 `action`、`expected_action`、KB 搜索结果和后 action 分支；
4. booking/change/cancel 的措辞是否会诱导当前 simulator 使用错误政策；
5. 所有样本仍只提供 CONTINUE，STOP 继续由当前 τ-bench 数据负责。

ABCD 与 AirDialogue 不应在第一轮实验中同时加入。先做单一来源 A/B，才能判断收益来自哪里。

## 6. 最低可接受的离线与在线指标

外部数据不能只看 SFT loss。至少比较同一 checkpoint、同一 seed 集合下的：

- 场景事实一致率、虚构标识符率；
- goal coverage / goal drift；
- progressive disclosure，而不是首轮倾倒全部字段；
- premature STOP、delayed STOP、termination F1；
- 对 Agent 错误、拒绝和澄清问题的恢复能力；
- τ-bench airline task success 与 pass^k；
- retail 跨域集上的行为退化，防止把 airline 模板学成通用策略。

只有主任务成功率不下降、STOP 校准不下降，并且目标保持或恢复能力有稳定提升，外部数据才应进入下一阶段。

## 7. 官方来源

- [ABCD 官方仓库（MIT，含数据结构）](https://github.com/asappresearch/abcd)
- [ABCD NAACL 2021 论文](https://aclanthology.org/2021.naacl-main.239/)
- [AirDialogue 官方仓库](https://github.com/google/airdialogue)
- [AirDialogue EMNLP 2018 论文](https://aclanthology.org/D18-1419/)
- [Google 官方 AirDialogue 数据卡（CC-BY-NC-4.0）](https://huggingface.co/datasets/google/air_dialogue)
- [Schema-Guided Dialogue 官方仓库（CC-BY-SA-4.0）](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- [MultiWOZ 官方仓库](https://github.com/budzianowski/multiwoz)
- [Taskmaster 官方仓库](https://github.com/google-research-datasets/Taskmaster)
- [Humanual-Chat 数据卡](https://huggingface.co/datasets/snap-stanford/humanual-chat)
- [RealUserSim 数据卡](https://huggingface.co/datasets/Salesforce/RealUserSim)
