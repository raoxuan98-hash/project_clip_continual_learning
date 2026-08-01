# launcher 进程计数 race condition 修复

**日期**: 2026-08-02
**会话概况**: 监控发现 supervisor 日志中 full-shot 进程数偶发误报为 8/22，修复 launcher 的 PID 过滤逻辑，改用一次性快照避免两次 pgrep 之间的 race。

---

## 1. 问题

- `scripts/pa_fullshot_launcher.sh` 与 `scripts/pa_wave_launcher.sh` 原先用以下方式判断顶层进程：
  ```bash
  pgrep -af "experiment_name PA__" | ... | while read pid; do
    ppid=$(awk '/PPid:/{print $2}' /proc/$pid/status)
    if ! pgrep -af "experiment_name PA__" | ... | grep -q "^${ppid}$"; then
      echo $pid
    fi
  done | wc -l
  ```
- 由于 DataLoader workers 在不断 fork/exit，两次 `pgrep` 调用之间的进程集合可能不一致，导致偶发把 worker 误判为顶层进程（日志出现 `22 / 2`、`8 / 2`）。
- 若误报持续 ≥2，supervisor 会在当前 run 结束后拒绝启动后续 LoRA-only full-shot。

## 2. 修复

- 新增一次性 PID 快照：
  ```bash
  _PA_PID_FILE=$(mktemp)
  trap 'rm -f "$_PA_PID_FILE"' EXIT
  _pa_all_pids() { pgrep -af "experiment_name PA__" | grep -v 'pgrep -af' | awk '{print $1}'; }
  ```
- `our_running` 与 `is_running` 都在同一份快照内做 PPid 查找，避免 race。
- 已验证：当前 2 个 full-shot 主进程被正确识别为 `2 / 2`。

## 3. 相关文件

- `scripts/pa_fullshot_launcher.sh`
- `scripts/pa_wave_launcher.sh`
- 已提交并推送：`df0f2b1`

## 4. 下一步

- 继续等待 LoRA-NF full-shot seed42/43 完成。
- 修复后的 launcher 将保证 supervisor 能正确调度 LoRA-only full-shot 与 B0。
