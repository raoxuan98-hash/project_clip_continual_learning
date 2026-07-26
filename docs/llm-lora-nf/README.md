# LLM LoRA-NF 开发文档

本目录保存 LoRA-NF 从 CLIP 扩展到 0.6B--3B 大语言模型的目标、执行约束和开发记录。

LLM 开发必须与当前 CLIP 代码隔离。目标已经用户批准；后续实现和服务器
实验仍须遵守 `GOAL.md` 的资源、镜像、证据和低存储约束。

2026-07-27 用户将全局 GPU 上限由 2 张调整为 3 张；每次准入至少留出
1 张真正空闲 GPU、无可用 GPU 时仅远程 CPU 跑链路的规则不变。当前
单个 TRACE runner 仍只请求 1 张卡，提高上限不会自动扩大单 run 占用。

## 当前状态

**状态：已批准；隔离实现与论文级门控已完成，本批远程测试通过。**

用户已于 2026-07-25 明确回复“开始执行”。当前已完成七方法统一链路、
Track A、checkpoint/export/evaluator、统计、正式配置硬门控与
content-addressed calibration cache；正式流水线另已加入“单最终 adapter +
单临时 merged checkpoint”低存储保留策略。2026-07-26 已在隔离 GPU
环境完成代码任务低存储编排后的 `105 passed`，并验证 RTX 4090 CUDA
前后向。
干净提交 `d817858` 的 Qwen3-0.6B 完整 LoRA-NF CPU smoke 也已通过；
可重建 adapter 验证后已删除，只保留约 16 KB 的非正式报告。

本轮静态审阅进一步固定了训练 seed 与 lm-eval seed 的边界、CPU smoke
禁用全量 request cache、共享 base evaluation 的内容寻址复用，以及
merged 槽位的跨进程流水线锁。随后又将标准 LoRA 基线切换为锁定的
PEFT 0.17.1，加入 CorDA 临时校准文件精确回收、LoRA-NF 二阶统计
PSD 硬校验，以及训练、评测、汇总三层源码/软件环境身份绑定；这些变更
已在服务器锁定依赖环境中通过完整测试。

代码任务现已固定 PiSSA 发布的 104,848 条 Python CodeFeedback、EvalPlus
v0.3.1、HumanEval+ v0.1.10、MBPP+ v0.2.0、greedy pass@1 和
Bubblewrap 分离执行链。数据与 evaluator 均位于服务器外部缓存，不进入
Git；基础链对应实现提交为
`006805ed39193d12406d242874f44e75c45b278c`，其后又补齐了串行低存储
launcher、训练 seed/adapter 哈希身份链和三 seed 配对统计。

TRACE 开发现已锁定官方 evaluator commit、两种任务顺序、epochs、
OP/BWT/forgetting 定义和正式数据资格门禁。TreeLoRA 固定 commit 的
TRACE-500 归档已通过 24 个 split 的逐文件行数、schema 与 SHA-256
校验，但明确只作为 pilot；它不会被冒充为每任务 5k/2k 的论文协议。
连续方法采用 task merge/reset：每个阶段只在内存追加可重放低秩分支，
最后写一个累计 adapter。LoRA-NF、LoRA-Null 与 DoRA 的多任务重放
等价测试均已通过，未引入八份阶段模型 checkpoint。TRACE 的数据读取、
Instruct chat 转换、确定性生成、八任务指标兼容层、三角任务×时间聚合和
连续训练 runner、状态消融编排和 token 长度审计也已接通；当前服务器
完整回归基线为 `137 passed`。长度审计进一步定位到当前训练编码的
右截断与 TRACE 官方左截断不一致：官方 LoRA 协议仍是
`1024 prompt + 512 answer = 1536` combined limit，但从左侧截断以保留
回答。修复现已进入服务器复核，不再把极长 MeetingBank prompt 误解为
必须将所有方法扩展到 48k 上下文。正式
5k/2k 数据尚未取得精确来源 manifest，因此 runner 会硬阻止正式结果，
只允许已核验 TRACE-500 做状态消融和链路 pilot。

干净提交 `2cdd550` 的 Qwen3-0.6B TRACE 最小 GPU 闭环已通过：只使用
GPU 0 并留空 GPU 5，完成完整 LoRA-NF 校准、一步训练、确定性生成、
指标聚合及累计 adapter 完整性/结构重载。输出硬标
`gpu_chain_smoke_only`，adapter 审计后已删除，只保留约 26 KB 报告、
预测和指标证据。

## 文件索引

- [`GOAL.md`](GOAL.md)：完整目标、范围、强制约束和完成标准。
- [`METHOD_SPEC.md`](METHOD_SPEC.md)：从当前 CLIP 实现审计得到的 LLM LoRA-NF 数学与实现规范。
- [`MODEL_DATA_EVAL_MANIFEST.md`](MODEL_DATA_EVAL_MANIFEST.md)：模型、数据、镜像、训练和评测版本清单。
- [`SERVER_RUNBOOK.md`](SERVER_RUNBOOK.md)：远程测试、gate、主矩阵、低存储评测和汇总命令。
- [`records/`](records/)：LLM 开发线的决策、实现、测试、实验和结果记录。
- [`records/2026-07-25-01-goal-definition.md`](records/2026-07-25-01-goal-definition.md)：目标形成过程和当前审批状态。
- [`records/2026-07-25-02-server-resource-policy.md`](records/2026-07-25-02-server-resource-policy.md)：GPU 上限、空闲 GPU 和 CPU 链路验证规则。
- [`records/2026-07-25-03-v4-review-decisions.md`](records/2026-07-25-03-v4-review-decisions.md)：自审后的方法位置、Instruct、Git 上传边界和可验收性修订。
- [`records/2026-07-25-04-phase-minus1-start.md`](records/2026-07-25-04-phase-minus1-start.md)：目标获批、方法审计和服务器首轮核验。
- [`records/2026-07-25-11-track-b-lora-null-calibration.md`](records/2026-07-25-11-track-b-lora-null-calibration.md)：Track B LoRA-Null 官方 calibration 修正。
- [`records/2026-07-25-12-cache-and-formal-protocol-guards.md`](records/2026-07-25-12-cache-and-formal-protocol-guards.md)：校准缓存、比较 hash 和正式配置硬门控。
- [`records/2026-07-26-01-formal-training-and-integrity-chain.md`](records/2026-07-26-01-formal-training-and-integrity-chain.md)：正式训练语义、环境锁与端到端完整性链。
- [`records/2026-07-26-02-checkpoint-retention-policy.md`](records/2026-07-26-02-checkpoint-retention-policy.md)：checkpoint 数量和低存储保留硬约束。
- [`records/2026-07-26-03-storage-baseline-and-provenance-audit.md`](records/2026-07-26-03-storage-baseline-and-provenance-audit.md)：PEFT LoRA 基线、缓存/临时文件回收、滤波统计安全和派生证据来源审计。
- [`records/2026-07-26-04-server-gpu-environment-and-clean-smoke.md`](records/2026-07-26-04-server-gpu-environment-and-clean-smoke.md)：可用 GPU 环境定位、完整测试、干净提交 CPU 闭环与 checkpoint 回收。
- [`records/2026-07-26-07-trace-data-and-cumulative-adapter.md`](records/2026-07-26-07-trace-data-and-cumulative-adapter.md)：TRACE 数据资格审计、指标定义与单最终累计 adapter 设计。
- [`records/2026-07-26-08-trace-runner-and-metrics.md`](records/2026-07-26-08-trace-runner-and-metrics.md)：TRACE Instruct 数据、确定性评测、连续 runner、低存储烟雾测试和完整回归。
- [`records/2026-07-27-01-gpu-limit-three.md`](records/2026-07-27-01-gpu-limit-three.md)：GPU 总上限由两张调整为三张，至少留一张与 CPU fallback 规则不变。
- [`records/2026-07-27-02-trace-length-audit.md`](records/2026-07-27-02-trace-length-audit.md)：状态 pilot 在第三阶段 MeetingBank 暴露右截断问题；全量审计与官方代码核对后锁定 1536 token combined-left 协议。

## 文档边界

- 本目录：专门记录 LLM LoRA-NF 开发线。
- `chat-history/`：继续保存项目级工程协作摘要，并链接本目录的详细记录。
- `chat-history-for-paper-writing/`：保存论文主张、章节组织和叙事变化。
- 现有 CLIP 文档与代码：在 LLM 目标获批前保持不变。
