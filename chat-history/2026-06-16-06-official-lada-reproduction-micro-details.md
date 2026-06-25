# 官方 LADA 复现中的训练微操记录

**日期**: 2026-06-16
**会话概况**: 在确认当前框架性能弱于 LADA 后，开始使用官方 `MaolinLuo/LADA` 代码做复现验证，并整理官方实现中会影响结果的关键训练细节，便于后续公平对比和整合。

---

## 1. 官方代码来源

远端官方仓库位置：

`/home/raoxuan/projects/LADA_official`

官方仓库：

`https://github.com/MaolinLuo/LADA`

当前复现 commit：

`c74806cd8ebe50f51dc6b448d3abe404ff4b41cb`

这不同于服务器上旧的 `/home/raoxuan/projects/LADA`。旧目录的 remote 指向 `lora-nsp-clip`，且工作区有本地改动，不能作为官方 LADA 复现依据。

---

## 2. 官方主配置

官方 16-shot 主配置来自：

- `configs/model/clip_vit_b16.yaml`
- `scripts/run_TAIL_16shot.sh`

关键参数：

- backbone: `CLIP-ViT-B/16`
- batch size: `64`
- precision: `amp`
- seed: `0`
- deterministic: `True`
- optimizer: `AdamW`
- learning rate: `0.001`
- weight decay: `5e-4`
- LADA lambda_1: `lada_k=16`
- DPT lambda_2: `prototype_k=4`
- DPT image prototype weight coefficient: `64.0`
- text-side PEFT: `t_adaptformer=True`
- text adapter dimension: `t_adapter_dim=16`

官方日志中第一任务参数量：

- Total params: `150655180`
- Tuned text encoder params: `215244`
- Tuned LADA params: `819200`

---

## 3. 优化对象

官方训练优化器只优化两类参数：

1. `self.model.text_tuner.parameters()`
2. `self.model.lada.curr_lada_features`

对应代码在官方 `trainer.py`：

```python
self.optim = torch.optim.AdamW(
    [
        {"params": self.model.text_tuner.parameters()},
        {"params": self.model.lada.curr_lada_features},
    ],
    lr=cfg.lr,
    weight_decay=cfg.weight_decay,
)
```

因此官方主流程不是图像主干微调。CLIP image encoder 在训练中作为特征提取器使用，主要可训练部分是文本侧 tuner 和当前任务新增的 LADA features。

这和我们当前 `main_incremental.py` 中的 LoRA-vision / LoRA-text 训练路径不是同一个训练假设。后续公平比较时，不能只对齐 classifier 或 metric，还需要对齐可训练参数范围。

---

## 4. 学习率退火

官方使用 `OneCycleLR`，不是 cosine，也不是固定学习率：

```python
self.sched = torch.optim.lr_scheduler.OneCycleLR(
    self.optim,
    max_lr=cfg.lr,
    epochs=cfg.num_epochs,
    steps_per_epoch=len(self.train_loader),
)
```

调度粒度是每个 batch 调一次：

```python
self.sched.step()
```

官方日志中可以观察到 OneCycleLR 行为。例如 aircraft 任务：

- epoch 1: lr 从约 `4e-5` 开始
- epoch 10-13: lr 接近 `1e-3`
- 后续逐步下降
- epoch 40: lr 下降到接近 0

这说明 LADA 的高性能不只是 LADA classifier 本身，还包含官方每任务 OneCycleLR 的训练 recipe。

---

## 5. 每数据集 epoch 不相同

官方 16-shot order-I 脚本不是每个任务统一训练步数，而是为每个数据集单独设置 epoch：

| dataset | official 16-shot epochs |
|---|---:|
| aircraft | 40 |
| caltech101 | 10 |
| dtd | 30 |
| eurosat | 100 |
| flowers | 30 |
| food101 | 5 |
| mnist | 200 |
| oxford_pets | 10 |
| stanford_cars | 30 |
| sun397 | 10 |

因为 16-shot 且 batch size 为 64，一个 epoch 的 batch 数大约是：

`ceil(num_classes * 16 / 64)`

实际例子：

- aircraft: `100` classes, `25` batches/epoch, `40` epochs -> `1000` optimizer steps
- caltech101: 约 `100` classes, 约 `25` batches/epoch, `10` epochs -> 约 `250` optimizer steps
- eurosat: `10` classes, 约 `3` batches/epoch, `100` epochs -> 约 `300` optimizer steps
- mnist: `10` classes, 约 `3` batches/epoch, `200` epochs -> 约 `600` optimizer steps
- stanford_cars: `196` classes, 约 `49` batches/epoch, `30` epochs -> 约 `1470` optimizer steps

这意味着官方 LADA 有明显的 per-dataset training budget tuning。若我们用统一 `iterations` 和它比较，训练预算并不严格对齐。

---

## 6. `lada_k` 与 `prototype_k` 的区别

官方配置中有两个容易混淆的 prototype 数量：

- `lada_k=16`
- `prototype_k=4`

它们分别对应论文中的：

- `lada_k`: `lambda_1`
- `prototype_k`: `lambda_2`

工作理解：

`lada_k` 是分类侧每类 LADA prototypes 的数量；`prototype_k` 是 DPT/GMM 回放侧每类 Gaussian components 的数量。

更精确地说：

### 6.1 `lada_k`: 每类分类原型数

`lada_k` 控制每个类别有多少个 LADA features / label-specific adapter vectors。

官方流程：

1. 对当前任务每个类别的训练图像提取 CLIP image features。
2. 对每个类别的 features 做 k-means。
3. 每个类别取 `lada_k` 个 cluster centers。
4. 这些 centers 成为当前任务的 LADA features。
5. 当前任务的 LADA features 在训练中是可学习参数。
6. 任务结束后，当前任务 LADA features 被冻结并保留。

这些 LADA features 直接参与分类 forward：

`image feature -> affinity to LADA features -> exp transform -> class logits`

因此 `lada_k` 是“分类器表达能力”相关的 prototype 数量。

以 Aircraft 为例：

- Aircraft 有 `100` 类。
- 官方 `lada_k=16`。
- 当前任务新增 LADA features 数量是 `100 * 16 = 1600`。
- CLIP feature dim 是 `512`。
- 因此当前任务可训练 LADA 参数量是 `100 * 16 * 512 = 819200`。

这与官方日志一致：

`Tuned LADA params: 819200`

### 6.2 `prototype_k`: 每类 DPT/GMM 分量数

`prototype_k` 控制每个类别保存多少个 DPT image prototype components，也可以理解为每类 GMM components 数量。

任务结束后，官方会为当前任务每个类别保存 `prototype_k` 个分布统计，通常包括：

- mean
- covariance / spherical variance
- mixture weight
- label

后续任务训练时，DPT 从这些历史分布分量中采样旧类 image prototypes，并把它们拼到当前任务 batch 前面，用来约束旧类决策边界。

因此 `prototype_k` 不是直接分类用的 LADA features，而是历史任务回放用的分布统计数量。

以 Aircraft 为例：

- Aircraft 有 `100` 类。
- 官方 `prototype_k=4`。
- 任务结束后保存的 image prototypes 数量是 `100 * 4 = 400`。

这与官方日志一致：

`Updated image prototypes shape: torch.Size([400, 512])`

到第二个约 100 类任务结束后，总 image prototypes 数量变为约：

`(100 + 100) * 4 = 800`

官方日志中也能观察到：

`Updated image prototypes shape: torch.Size([800, 512])`

### 6.3 简洁总结

| 参数 | 论文符号 | 工作含义 | 用途 | 是否直接参与分类 | 是否可训练 |
|---|---|---|---|---|---|
| `lada_k` | `lambda_1` | 每类分类原型数 | 构建 LADA features | 是 | 当前任务可训练，旧任务冻结 |
| `prototype_k` | `lambda_2` | 每类 GMM 分量数 | DPT 历史回放 | 否，作为训练约束样本 | 否，保存统计量 |

一句话：

`lambda_1 = 每类分类原型数；lambda_2 = 每类历史分布高斯分量数。`

这个区别对后续整合很重要。不能把 `prototype_k` 简单当成 LR-RGDA 里的每类 classifier centers，也不能把 `lada_k` 当成 DPT 的 GMM 分量数。官方 LADA 中，二者服务于不同目标：

- `lada_k` 增强当前/已见类的分类表达能力。
- `prototype_k` 低成本保存历史类分布，用于后续任务训练时的 DPT 约束。

---

## 7. DPT 与第一任务特殊处理

官方脚本第一任务使用：

`continue_train_first True`

这使第一任务不会回放旧任务原型，因为还没有历史任务。后续任务默认 `continue_train=True` 且 `continue_train_first=False`，会使用 DPT image prototypes。

训练中如果不是第一任务，官方会：

1. 从历史 image prototypes 的 GMM 统计中采样增强原型；
2. 将采样的 image prototypes 拼接到当前 batch 的 image features 前；
3. 拼接历史 text prototypes 和当前任务 text features；
4. 用 prototype weights 乘以 `image_prototypes_weight_coef=64.0`；
5. 对拼接后的 seen-class 分类空间计算 CE。

这部分是官方 LADA 的稳定性来源之一，不能用“从 GMM 重新构建一个普通 classifier”来等价替代。

---

## 8. 子集任务验证的注意事项

我们启动了官方两任务验证：

- dataset sequence: `aircraft -> caltech101`
- output dir: `TAIL_16shot_two_official_20260616`
- official epochs: `aircraft=40`, `caltech101=10`

两任务结果量级基本对上，尤其是：

- aircraft 当前任务约 `48-49%`
- caltech101 最终约 `93-96%`

但必须注意：官方代码的 `dataset_sequence` 不只是训练顺序，也用于构建 task-agnostic X-TAIL testloader 和全局 class space。

因此两任务验证的第一行 `caltech101` accuracy 会高于 README 10 任务示例中的 `75.0`，因为两任务 run 的候选类别空间只包含 `aircraft + caltech101`，而完整 10 任务 run 的候选类别空间包含全部 10 个数据集。减少任务数会降低 task-agnostic classification 难度。

结论：

- 两任务 run 适合验证官方代码能跑、数据路径正确、训练 recipe 生效。
- 两任务 run 不能直接用于复现论文主表数值。
- 要验证论文前几列的 Transfer/Average/Last，应至少用包含这些未来任务的 `dataset_sequence`，最好完整 10 任务。

---

## 9. 四任务验证

为进一步验证官方实现，我们启动了四任务官方 order-I 子集：

`aircraft -> caltech101 -> dtd -> eurosat`

输出目录：

`/home/raoxuan/projects/LADA_official/output/TAIL_16shot_four_official_20260616`

日志目录：

`/home/raoxuan/projects/LADA_official/run_logs/TAIL_16shot_four_official_20260616_*.log`

四任务使用官方对应 epoch：

- aircraft: `40`
- caltech101: `10`
- dtd: `30`
- eurosat: `100`

启动时确认配置：

- root: `/data1/open_datasets/X-TAIL`
- dataset_sequence: `['aircraft', 'caltech101', 'dtd', 'eurosat']`
- num_shots: `16`
- batch_size: `64`
- lr: `0.001`
- scheduler: `OneCycleLR`
- output_dir: `TAIL_16shot_four_official_20260616`

该 run 仍然是子集验证。它比两任务更接近主表前四列，因为前两步会在四数据集候选空间上评估 `dtd/eurosat`，但它仍不等价于完整 10 任务结果。

四任务 run 已完成。由于官方 `result_process.py` 固定读取完整 10 任务 `TAIL.yaml` 并寻找默认日志名，而本次是四任务子集和自定义日志名，因此用日志中的 `* Task k Accuracy` 手动解析矩阵。

四任务矩阵如下，行表示训练到哪个任务后，列表示测试任务：

| train step | aircraft | caltech101 | dtd | eurosat |
|---|---:|---:|---:|---:|
| aircraft | 48.2 | 80.2 | 43.5 | 39.1 |
| caltech101 | 48.8 | 92.9 | 43.3 | 38.9 |
| dtd | 48.9 | 93.6 | 70.9 | 34.8 |
| eurosat | 48.9 | 93.6 | 71.4 | 87.9 |

四任务子集指标：

| metric | aircraft | caltech101 | dtd | eurosat | mean |
|---|---:|---:|---:|---:|---:|
| Transfer | N/A | 80.2 | 43.4 | 37.6 | 53.7 |
| Average | 48.7 | 90.1 | 57.3 | 50.2 | 61.6 |
| Last | 48.9 | 93.6 | 71.4 | 87.9 | 75.5 |

和官方 README 的完整 10 任务示例前四列相比，方向和量级基本一致：

| train step | README aircraft | README caltech101 | README dtd | README eurosat |
|---|---:|---:|---:|---:|
| aircraft | 48.3 | 75.0 | 36.4 | 37.4 |
| caltech101 | 48.8 | 91.6 | 35.8 | 37.2 |
| dtd | 48.8 | 92.5 | 66.6 | 33.6 |
| eurosat | 48.8 | 92.5 | 66.6 | 86.9 |

四任务子集中 `caltech101/dtd` 的部分列更高，主要因为候选类别空间只有四个数据集，而不是完整 10 个数据集。这个结果仍然支持一个明确判断：官方 LADA 代码、数据路径、训练 recipe 和性能量级是可信的，LADA 确实是一个强基线。

---

## 10. 文本编码器微调与 zero-shot classifier 的共享方式

官方 LADA 的文本侧机制有一个容易忽略的细节：seen classes 和 unseen classes 的 text features 来源不同。

训练阶段：

- 当前任务 prompts 会通过 `self.model.text_encoder(self.model.prompts, self.model.text_tuner)` 得到当前任务 text features。
- 默认 `text_tuner` 是 text-side AdaptFormer，配置为 `t_adaptformer=True, t_adapter_dim=16`。
- 训练 loss 使用 text logits + LADA logits。
- 任务结束后，官方调用：

```python
self.model.dpt.update_text_prototypes(self.model.prompts, self.model.text_tuner)
```

这会把当前任务经过训练后 text_tuner 编码得到的 text features 追加到：

```python
self.model.dpt.text_prototypes
```

因此，历史 seen classes 的 text features 是“任务结束时固化保存”的 text prototypes，而不是后续任务中重新用当前 text_tuner 动态编码。

评估阶段 `test_wo_selector()`：

1. 加载一个 frozen original CLIP，构建完整 `merged_classnames` 的 zero-shot text features。
2. 对 seen classes，使用训练过程中保存下来的 `self.model.dpt.text_prototypes`。
3. 对 unseen classes，使用 frozen CLIP 的 zero-shot text features。
4. 拼接：

```python
origin_text_features = frozen_clip.text_features
id_text_features = self.model.dpt.text_prototypes
id_num = id_text_features.shape[0]
ood_text_features = origin_text_features[id_num:, :]
text_features = torch.cat((id_text_features, ood_text_features), dim=0)
```

然后用这个拼接后的 text classifier 与 LADA features 一起评估：

```python
output = self.model(
    image,
    text_features=text_features,
    lada_features=self.model.lada.prev_lada_features,
    classifier=self.model.lada.joint_classifier,
)
```

这说明官方 LADA 的 `LADA + zero-shot` 不是简单地在所有类别上使用 frozen zero-shot classifier：

- seen classes: 使用训练后保存的 text prototypes；
- unseen classes: 使用 frozen CLIP zero-shot classifier；
- seen classes 还额外叠加 LADA logits。

这种机制会让 text-side tuning 的收益被固化到历史任务 text prototypes 中，同时避免未来 unseen classes 被 text_tuner 破坏。

### 10.1 关闭文本调参时的含义

若通过配置关闭 text-side AdaptFormer：

`t_adaptformer=False`

并且不打开其他 text tuning 选项，则 `Text_Tuner` 变成空参数模块：

- official text_tuner params: `215244`
- no-text-tuning text_tuner params: `0`

此时：

- 当前任务 text features 退化为 frozen CLIP prompt features；
- `dpt.text_prototypes` 保存的是 frozen CLIP 的 seen-class text features；
- optimizer 实际只优化 `self.model.lada.curr_lada_features`；
- 评估时 seen classes 和 unseen classes 都来自 frozen CLIP text features，但 seen classes 会额外叠加 LADA logits。

这才是比较干净的“只训练 LADA features + 使用 frozen zero-shot classifier”的实验设置。

计划运行四任务对照：

- dataset sequence: `aircraft -> caltech101 -> dtd -> eurosat`
- output dir: `TAIL_16shot_four_official_no_text_20260616`
- 与官方四任务 run 保持相同 epoch 和其他超参
- 唯一核心差异：`t_adaptformer=False`

这个实验用于回答：在没有文本编码器微调的情况下，单独 LADA features 加 frozen zero-shot classifier 能保留多少性能。

### 10.2 四任务 no-text-tuning 对照结果

no-text-tuning 对照已完成。配置确认：

- output dir: `TAIL_16shot_four_official_no_text_20260616`
- `t_adaptformer=False`
- `Tuned text encoder params: 0`
- `Tuned LADA params: 819200` on aircraft
- 其他训练 recipe 与官方四任务 run 保持一致

完整矩阵：

| train step | aircraft | caltech101 | dtd | eurosat |
|---|---:|---:|---:|---:|
| aircraft | 44.4 | 78.3 | 43.4 | 39.1 |
| caltech101 | 44.4 | 82.3 | 43.4 | 39.1 |
| dtd | 44.4 | 86.0 | 65.5 | 39.1 |
| eurosat | 44.4 | 86.1 | 67.1 | 86.3 |

四任务子集指标：

| metric | aircraft | caltech101 | dtd | eurosat | mean |
|---|---:|---:|---:|---:|---:|
| Transfer | N/A | 78.3 | 43.4 | 39.1 | 53.6 |
| Average | 44.4 | 83.2 | 54.9 | 50.9 | 58.3 |
| Last | 44.4 | 86.1 | 67.1 | 86.3 | 71.0 |

与官方 text-tuned 四任务 run 的差异，即 `no_text - official_text_tuned`：

| train step | aircraft | caltech101 | dtd | eurosat |
|---|---:|---:|---:|---:|
| aircraft | -3.8 | -1.9 | -0.1 | +0.0 |
| caltech101 | -4.4 | -10.6 | +0.1 | +0.2 |
| dtd | -4.5 | -7.6 | -5.4 | +4.3 |
| eurosat | -4.5 | -7.5 | -4.3 | -1.6 |

均值差异：

- Transfer Mean: `-0.1`
- Average Mean: `-3.2`
- Last Mean: `-4.5`

初步解释：

1. **Transfer 几乎不变**：Transfer 主要看未来 unseen tasks。在官方 `test_wo_selector()` 中，unseen classes 本来就使用 frozen CLIP zero-shot text features，所以关闭 text_tuner 对未来任务列影响有限。
2. **Average 和 Last 明显下降**：seen/current tasks 更依赖训练后的 text prototypes 与 LADA features 的组合。关闭 text_tuner 后，seen-class text prototypes 退化为 frozen CLIP text features，只剩 LADA features 可学习，因此 learned-task accuracy 降低。
3. **Caltech101 降幅最大**：最终 caltech101 从 `93.6` 降到 `86.1`，说明官方 text-side AdaptFormer 对 seen-class refinement 很关键。
4. **DPT/LADA 单独仍然有效**：no-text-tuning 的 Last Mean 仍有 `71.0`，说明 LADA features + DPT + frozen zero-shot classifier 本身也提供了较强能力，但不足以达到完整官方 LADA 的 `75.5` 四任务 Last Mean。

结论：官方 LADA 的性能来自两部分叠加：

- LADA/DPT 提供主要的 label-specific feature 和历史分布约束；
- text-side AdaptFormer 进一步显著提升 seen/current task 的 Average 与 Last。

这也解释了为什么官方会在历史任务中保存 `dpt.text_prototypes`：text-side tuning 的收益被固化到 seen classes，而未来 unseen classes 仍保持 frozen CLIP zero-shot classifier。

### 10.3 无 LADA logits、仅文本侧微调对照结果

为了进一步拆分官方 LADA 的收益，我们又补充了一组对照：保留官方 text-side AdaptFormer，但关闭 LADA logits。

具体做法是设置：

`alpha=0.0`

官方 forward 中最终 logits 是 text logits 和 LADA logits 的加权组合。将 `alpha` 置为 `0.0` 后，LADA logits 在输出中被置零，因此行为上等价于“不使用 LADA 分类器 logits”，只评估文本侧微调、历史 text prototypes、DPT replay 训练约束以及 frozen zero-shot unseen classifier 的组合效果。

需要注意一个 caveat：官方代码仍然会构建 LADA features，并且 optimizer 参数组里仍包含 `model.lada.curr_lada_features`。但由于 LADA logits 被 `alpha=0.0` 乘掉，LADA features 对最终 loss 应该没有有效梯度贡献。因此这是一个行为上较干净的 “no-LADA-logits” 对照，但不是代码结构上完全移除 LADA 模块的最小实现。

配置确认：

- output dir: `TAIL_16shot_four_official_text_only_alpha0_20260617`
- `alpha=0.0`
- `t_adaptformer=True`
- `Tuned text encoder params: 215244`
- `Tuned LADA params: 819200` on aircraft, but LADA logits are disabled by `alpha=0.0`
- 其他训练 recipe 与官方四任务 run 保持一致

完整矩阵：

| train step | aircraft | caltech101 | dtd | eurosat |
|---|---:|---:|---:|---:|
| aircraft | 45.9 | 80.5 | 43.5 | 39.1 |
| caltech101 | 47.4 | 88.6 | 43.1 | 39.0 |
| dtd | 47.4 | 90.4 | 69.0 | 35.4 |
| eurosat | 47.4 | 90.5 | 69.7 | 87.9 |

四任务子集指标：

| metric | aircraft | caltech101 | dtd | eurosat | mean |
|---|---:|---:|---:|---:|---:|
| Transfer | N/A | 80.5 | 43.3 | 37.8 | 53.9 |
| Average | 47.0 | 87.5 | 56.3 | 50.4 | 60.3 |
| Last | 47.4 | 90.5 | 69.7 | 87.9 | 73.9 |

与完整官方四任务 run 的差异，即 `text_only_alpha0 - official_full`：

| train step | aircraft | caltech101 | dtd | eurosat |
|---|---:|---:|---:|---:|
| aircraft | -2.3 | +0.3 | +0.0 | +0.0 |
| caltech101 | -1.4 | -4.3 | -0.2 | +0.1 |
| dtd | -1.5 | -3.2 | -1.9 | +0.6 |
| eurosat | -1.5 | -3.1 | -1.7 | +0.0 |

均值差异：

- Transfer Mean: `+0.1`
- Average Mean: `-1.3`
- Last Mean: `-1.6`

与 no-text-tuning / LADA-only 对照相比，即 `text_only_alpha0 - no_text_lada_only`：

- Transfer Mean: `+0.3`
- Average Mean: `+2.0`
- Last Mean: `+2.9`

三组四任务结果可以这样拆分：

| setting | text tuner | LADA logits | Transfer Mean | Average Mean | Last Mean |
|---|---|---|---:|---:|---:|
| full official LADA | yes | yes | 53.7 | 61.6 | 75.5 |
| no-text-tuning / LADA-only | no | yes | 53.6 | 58.3 | 71.0 |
| text-only alpha=0 | yes | no | 53.9 | 60.3 | 73.9 |

初步解释：

1. **文本侧微调本身贡献很大**：从 no-text-tuning 的 `Average 58.3 / Last 71.0` 到 text-only alpha=0 的 `Average 60.3 / Last 73.9`，说明官方 text-side AdaptFormer + cached text prototypes 已经解释了相当多的增益。
2. **LADA logits 仍然有额外收益**：从 text-only alpha=0 到 full official LADA，`Average +1.3`，`Last +1.6`。这说明 LADA 分类 logits 不是唯一收益来源，但确实进一步提升 seen/current tasks。
3. **Transfer 仍基本不受影响**：三组 Transfer Mean 都在 `53.6-53.9`。这继续支持前面的判断：在官方 `test_wo_selector()` 中，未来 unseen classes 主要由 frozen CLIP zero-shot features 决定。
4. **官方强性能是组合 recipe 的结果**：仅说“LADA classifier 很强”并不完整。更准确地说，官方 LADA 的四任务强性能来自 text-side AdaptFormer、cached seen-class text prototypes、DPT replay、LADA logits、OneCycleLR 和 per-dataset training budget 的组合。

---

## 11. 对我们框架整合的启发

目前能确认的核心点：

1. LADA 的高性能不是单一模块效果，而是官方 recipe 的组合结果。
2. 需要同时对齐：
   - 可训练参数范围；
   - batch size；
   - per-dataset epoch / step budget；
   - OneCycleLR；
   - first-task handling；
   - DPT prototype sampling 与 weighting；
   - task-agnostic global class space；
   - LADA metric 的矩阵定义。
3. 如果只把 LADA classifier 逻辑嫁接到我们的 `main_incremental.py`，但保留我们的统一 iterations、LoRA 训练路径、scheduler 和评估子集，结果不能代表官方 LADA。
4. 官方 text-side tuning 的收益通过 `dpt.text_prototypes` 固化进历史 seen classes；这和我们当前每次动态构造 zero-shot classifier 的方式不同。
5. 单独关闭 LADA logits 后，text-only alpha=0 仍达到 `Average 60.3 / Last 73.9`，说明后续整合时不能把提升全部归因于 LADA classifier；文本侧 AdaptFormer 和 cached text prototypes 需要单独建模、单独消融。
6. 后续若要做公平整合，建议先实现一个 `official_lada_recipe` 或类似配置模式，把以上 recipe 作为整体对齐，而不是逐项零散改动。

---

## 12. 下一步建议

1. 等四任务官方 run 完成，读取 `result.txt` 或用 `result_process.py` 汇总矩阵。
2. 将四任务矩阵与 README/论文 10 任务矩阵的前四列做方向性比较，但不要声称完全复现主表。
3. 若需要更严格的 text-only 对照，可以进一步 patch 官方代码，让 optimizer 不包含 `curr_lada_features`，并跳过 LADA feature 构建；当前 `alpha=0.0` 对照已经足够回答“无 LADA logits 时文本侧微调有多少收益”。
4. 若四任务无异常，启动完整官方 `scripts/run_TAIL_16shot.sh`，只覆盖 `root` 和 GPU 绑定，不改官方训练 recipe。
5. 在我们框架中新增一个 LADA recipe audit 表，逐项记录和官方差异，避免后续比较时误把 recipe 差异解释成方法差异。
