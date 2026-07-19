# LADA 离线权重修复（14:35-14:46）

**日期**: 2026-07-19
**会话概况**: LADA 16-shot 首启失败（clip 权重下载 DNS 失败），经转换脚本与官方重下两路解决。

## 1. 故障
- LADA 官方代码  需 OpenAI JIT 权重（~/.cache/clip/ViT-B-16.pt），本机缓存为空且首次下载 DNS 解析失败（rc=1，3 runs 全崩于 aircraft）。

## 2. 解决
- 编写 scripts/convert_hf_to_openai_clip.py：HF safetensors → OpenAI 命名 state_dict（in_proj q/k/v 拼接、投影转置、ln/mlp 映射），经 LADA build_model 严格加载 + HF 特征比对（cos=1.0）验证。
- 保存后官方代码因 SHA256 不匹配触发重下，此次下载成功（网络抖动恢复），最终使用官方原版 351MB 权重。
- LADA 16-shot ×3 seeds 于 14:45 正常进入训练（aircraft epoch 10/40 观测点）。

## 3. 备注
- 转换脚本保留为离线后备方案；若服务器外网再次不可用，重新生成同名文件并临时屏蔽 _download 的 sha256 校验。
