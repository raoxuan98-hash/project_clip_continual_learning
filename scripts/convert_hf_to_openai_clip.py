#!/usr/bin/env python3
"""将 HF transformers CLIP (clip-vit-base-patch16) 权重转换为 OpenAI JIT 命名的
state_dict，保存为 ~/.cache/clip/ViT-B-16.pt，供 LADA 官方代码（load_clip_to_cpu
的 torch.load 回退路径 + clip.build_model）离线加载。

转换后用 LADA_official 的 clip.build_model 严格 load_state_dict 验证，
并与 HF 模型在同一输入上比对特征余弦相似度。

用法（服务器）：
    python scripts/convert_hf_to_openai_clip.py \
        --hf-dir /mnt/raoxuan/models/clip-vit-base-patch16 \
        --lada-clip-dir /home/raoxuan/projects/LADA_official \
        --out ~/.cache/clip/ViT-B-16.pt
"""
import argparse
import sys
from pathlib import Path

import torch


def convert_block(sd, hf_prefix, oai_prefix, out):
    """转换一个 transformer layer 的注意力/MLP/LN 参数。"""
    q = sd[f"{hf_prefix}.self_attn.q_proj.weight"]
    k = sd[f"{hf_prefix}.self_attn.k_proj.weight"]
    v = sd[f"{hf_prefix}.self_attn.v_proj.weight"]
    qb = sd[f"{hf_prefix}.self_attn.q_proj.bias"]
    kb = sd[f"{hf_prefix}.self_attn.k_proj.bias"]
    vb = sd[f"{hf_prefix}.self_attn.v_proj.bias"]
    out[f"{oai_prefix}.attn.in_proj_weight"] = torch.cat([q, k, v], dim=0)
    out[f"{oai_prefix}.attn.in_proj_bias"] = torch.cat([qb, kb, vb], dim=0)
    out[f"{oai_prefix}.attn.out_proj.weight"] = sd[f"{hf_prefix}.self_attn.out_proj.weight"]
    out[f"{oai_prefix}.attn.out_proj.bias"] = sd[f"{hf_prefix}.self_attn.out_proj.bias"]
    out[f"{oai_prefix}.ln_1.weight"] = sd[f"{hf_prefix}.layer_norm1.weight"]
    out[f"{oai_prefix}.ln_1.bias"] = sd[f"{hf_prefix}.layer_norm1.bias"]
    out[f"{oai_prefix}.mlp.c_fc.weight"] = sd[f"{hf_prefix}.mlp.fc1.weight"]
    out[f"{oai_prefix}.mlp.c_fc.bias"] = sd[f"{hf_prefix}.mlp.fc1.bias"]
    out[f"{oai_prefix}.mlp.c_proj.weight"] = sd[f"{hf_prefix}.mlp.fc2.weight"]
    out[f"{oai_prefix}.mlp.c_proj.bias"] = sd[f"{hf_prefix}.mlp.fc2.bias"]
    out[f"{oai_prefix}.ln_2.weight"] = sd[f"{hf_prefix}.layer_norm2.weight"]
    out[f"{oai_prefix}.ln_2.bias"] = sd[f"{hf_prefix}.layer_norm2.bias"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="/mnt/raoxuan/models/clip-vit-base-patch16")
    ap.add_argument("--lada-clip-dir", default="/home/raoxuan/projects/LADA_official")
    ap.add_argument("--out", default=str(Path.home() / ".cache/clip/ViT-B-16.pt"))
    args = ap.parse_args()

    from transformers import CLIPModel
    hf = CLIPModel.from_pretrained(args.hf_dir, use_safetensors=True, local_files_only=True)
    sd = hf.state_dict()
    n_layers_text = hf.config.text_config.num_hidden_layers
    n_layers_vis = hf.config.vision_config.num_hidden_layers

    out = {}
    # 顶层
    out["logit_scale"] = sd["logit_scale"]
    out["token_embedding.weight"] = sd["text_model.embeddings.token_embedding.weight"]
    out["positional_embedding"] = sd["text_model.embeddings.position_embedding.weight"]
    out["ln_final.weight"] = sd["text_model.final_layer_norm.weight"]
    out["ln_final.bias"] = sd["text_model.final_layer_norm.bias"]
    out["text_projection"] = sd["text_projection.weight"].t()
    out["visual.class_embedding"] = sd["vision_model.embeddings.class_embedding"]
    out["visual.conv1.weight"] = sd["vision_model.embeddings.patch_embedding.weight"]
    out["visual.positional_embedding"] = sd["vision_model.embeddings.position_embedding.weight"]
    out["visual.ln_pre.weight"] = sd["vision_model.pre_layrnorm.weight"]
    out["visual.ln_pre.bias"] = sd["vision_model.pre_layrnorm.bias"]
    out["visual.ln_post.weight"] = sd["vision_model.post_layernorm.weight"]
    out["visual.ln_post.bias"] = sd["vision_model.post_layernorm.bias"]
    out["visual.proj"] = sd["visual_projection.weight"].t()
    # 文本层
    for i in range(n_layers_text):
        convert_block(sd, f"text_model.encoder.layers.{i}", f"transformer.resblocks.{i}", out)
    # 视觉层
    for i in range(n_layers_vis):
        convert_block(sd, f"vision_model.encoder.layers.{i}", f"visual.transformer.resblocks.{i}", out)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"saved {len(out)} tensors -> {out_path}")

    # ---- 验证 1：LADA build_model 严格加载 ----
    sys.path.insert(0, args.lada_clip_dir)
    from clip.model import build_model  # LADA fork 的 OpenAI CLIP
    loaded = torch.load(out_path, map_location="cpu")
    model = build_model(loaded)
    model.float().eval()
    print("✓ LADA clip.build_model strict load OK")

    # ---- 验证 2：与 HF 模型特征一致性 ----
    from transformers import CLIPProcessor
    from PIL import Image
    import numpy as np
    processor = CLIPProcessor.from_pretrained(args.hf_dir, local_files_only=True, use_fast=True)
    hf.float().eval()

    img = Image.fromarray((np.random.RandomState(0).rand(224, 224, 3) * 255).astype("uint8"))
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        f_hf = hf.get_image_features(**inputs)
        f_hf = f_hf / f_hf.norm(dim=-1, keepdim=True)
        f_oa = model.encode_image(inputs["pixel_values"])
        f_oa = f_oa / f_oa.norm(dim=-1, keepdim=True)
        texts = ["a photo of a cat", "a photo of an aircraft"]
        tin = processor(text=texts, return_tensors="pt", padding=True)
        t_hf = hf.get_text_features(**tin)
        t_hf = t_hf / t_hf.norm(dim=-1, keepdim=True)
        # LADA tokenizer（simple_tokenizer），需手动加 sot/eot（encode_text 按 eot 位置取特征）
        from clip.simple_tokenizer import SimpleTokenizer
        tok = SimpleTokenizer()
        sot = tok.encoder["<|startoftext|>"]
        eot = tok.encoder["<|endoftext|>"]
        padded = torch.zeros(2, 77, dtype=torch.long)
        for j, t in enumerate(texts):
            ids = [sot] + tok.encode(t) + [eot]
            padded[j, : len(ids)] = torch.tensor(ids)
        t_oa = model.encode_text(padded)
        t_oa = t_oa / t_oa.norm(dim=-1, keepdim=True)
    img_cos = (f_hf @ f_oa.t()).item()
    txt_cos = (t_hf @ t_oa.t()).diag()
    print(f"image feature cosine(HF, OpenAI) = {img_cos:.6f}")
    print(f"text feature cosine = {[round(v, 6) for v in txt_cos.tolist()]}")
    assert img_cos > 0.999, "image features mismatch"
    assert all(v > 0.999 for v in txt_cos.tolist()), "text features mismatch"
    print("✓ 特征一致性验证通过（cos > 0.999）")


if __name__ == "__main__":
    main()
