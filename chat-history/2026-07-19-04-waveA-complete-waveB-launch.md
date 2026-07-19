# Wave A 完成 + Wave B 启动（08:32-09:09）

**日期**: 2026-07-19
**会话概况**: 巡检 cron：Wave A 12/12 验收通过，Wave B 9 runs 启动；记录一个编辑运行中脚本事故的根因。

---

## 1. Wave A 验收（终验）

- 12/12 runs 全部 rc=0 完成，`audit_campaign_run.py --require-reference` 12/12 PASS。
- 配置/变体覆盖：lora_nf ×3、lora ×3、lora_null ×3、gradproj ×3（seed 42/43/44）。

## 2. 事故记录：启动器语法错误（无实际损失）

- 现象：6 个 waveA 启动器在完成全部 run 后报 `rerun_campaign.sh: line 163: syntax error`。
- 根因：04:05 提交 d323470（waveC_lada 分支）时，6 个启动器正在运行；bash 按字节偏移增量读脚本，文件变更导致续读错位。**教训：启动器运行期间不得修改 rerun_campaign.sh；后续 wave 的脚本修改必须在全部启动器结束后进行，或先复制为新文件再改。**
- 影响：仅启动器收尾阶段报错，12/12 结果完整；当前脚本 bash -n 通过。

## 3. Wave B 启动（09:08:55）

- 9 runs：C0(fd0,cd0) ×3 seeds（GPU 0-2）、C1(fd1,cd0) ×3（GPU 3-5）、C3(fd1,cd2) ×3（GPU 0-2 第二顺位）。
- C2(fd0,cd2) 复用 Wave A lora_nf 结果，不重复跑。
- 预计 ~12:45 完成；巡检 cron 更新为 4921204a（waveB 对象，含 Wave C 两段接力指令）。
