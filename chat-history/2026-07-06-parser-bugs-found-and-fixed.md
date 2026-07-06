# Pre-Wave A 启动前代码审查：Parser 参数缺失发现

**日期**: 2026-07-06
**会话**: Pre-Wave A 启动前的代码审查

---

## 发现

在审查 `main_incremental.py` 的 argument parser 时，发现以下参数缺失：

### 1. `--cd_divergence` 和 `--cd_temperature` 未定义

`src/trainers/lora_nsp_trainer.py` 中通过 `getattr` 访问这两个参数：

```python
cd_divergence = getattr(self.args, "cd_divergence", "kl_forward")
cd_temperature = getattr(self.args, "cd_temperature", 2.0)
```

但 `main_incremental.py` 的 `ArgumentParser` 从未注册这两个 flag。任何脚本若传入 `--cd_temperature 4.0` 或 `--cd_divergence kl_reverse`，argparse 会直接报错退出。

经 git log 追溯（`git show 06b440d:main_incremental.py`），在本次修复前的最新 commit 中这两个参数均不存在。推测在某次代码清理（commit `06b440d "clean"`）中被意外移除。

### 2. `--scheduler` choices 不完整

Trainer (`lora_nsp_trainer.py:455-481`) 支持 5 种 scheduler：

```python
cosine, onecycle, linear, constant, cosine_with_warmup
```

但 CLI choices 只列了 `["cosine", "onecycle"]`。传入 `--scheduler cosine_with_warmup` 会直接被 argparse 拒绝。

### 影响范围

| 6-task 消融实验 | 依赖的参数 | 是否会受影响 |
|------|------|:---:|
| CD 散度形式（§5.6, 6 种 kl_forward/kl_reverse/js/mse/cosine/l1） | `--cd_divergence` | **受影响** |
| CD 温度（§5.7, temp=1.0/2.0/4.0） | `--cd_temperature` | **受影响** |
| Iter × Scheduler（§5.8, cosine_with_warmup/linear/constant） | `--scheduler` | **受影响** |

> **注意**：尽管参数缺失，这些实验的结果文件确实存在于服务器 `experiments/optimizer_ablation/` 目录下（含 `schedcosine_with_warmup`、`schedconstant`、`schedlinear` 等），推测实验在参数被移除前已完成。在本次修复前，若用当前代码重跑这些实验，将全部崩溃。

### 修复

在 `main_incremental.py` 中（commit `710ee34`）：

1. 新增 `--cd_divergence`（choices: kl_forward, kl_reverse, js, mse, cosine, l1, default: kl_forward）
2. 新增 `--cd_temperature`（type: float, default: 2.0）
3. 扩展 `--scheduler` choices: `["cosine", "onecycle", "cosine_with_warmup", "linear", "constant"]`

同时确认代码使用 `parser.parse_args()`（非 `parse_known_args()`），未定义参数会正确报错。

### 对 10-task 重消融的影响

若不修复，以下 Waves 的实验将无法启动：

| Wave | 依赖参数 | 状态 |
|------|---------|:---:|
| Pre-A | `cosine` | ✅ 不受影响 |
| Wave A | `cosine_with_warmup` | 🔴 会崩溃 |
| Wave D | `--cd_temperature` | 🔴 会崩溃 |
| Wave F | `freeze_after/low_lr_after/always/never` | ✅ 不受影响 |

### 教训

- 代码清理时应检查 `getattr` 调用是否依赖未注册的 CLI 参数
- 启动脚本中的参数应全部在 parser 中显式注册，不依赖 `getattr` 的 fallback
- 未来重构时建议用 `grep -r "getattr.*args"` 交叉验证 parser 注册表

---

## 相关文件

- `main_incremental.py:511-513` — scheduler choices 修复
- `main_incremental.py:580-583` — cd_divergence / cd_temperature 新增
- `src/trainers/lora_nsp_trainer.py:455-481` — trainer scheduler 处理
- `src/trainers/lora_nsp_trainer.py:625-626` — cd_divergence / cd_temperature getattr
- `scripts/run_cd_divergence_ablation.sh` — 使用受影响参数的旧启动脚本
