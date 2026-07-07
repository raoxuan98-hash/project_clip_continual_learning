import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.utils.infinite_sampler import InfiniteSampler

class MergedReferenceDataset(Dataset):
    """
    合并的参考数据集，用于蒸馏。

    支持两种模式：
    - tokenized=False（默认）：返回 4 元组 (images, texts, img_features, txt_features)
    - tokenized=True：返回 5 元组 (images, input_ids, attention_mask, img_features, txt_features)，
      避免训练循环每步重复调用 tokenizer。
    """
    def __init__(self, images, text_data, img_features, txt_features, tokenized=False):
        self.images = images
        self.tokenized = tokenized
        if tokenized:
            self.text_input_ids, self.text_attention_mask = text_data
        else:
            self.texts = text_data
        self.img_features = img_features
        self.txt_features = txt_features

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.tokenized:
            return (self.images[idx],
                    self.text_input_ids[idx],
                    self.text_attention_mask[idx],
                    self.img_features[idx],
                    self.txt_features[idx])
        else:
            return (self.images[idx],
                    self.texts[idx],
                    self.img_features[idx],
                    self.txt_features[idx])

def load_reference_dataset(args, model_pretrain, processor, device,
                           return_tokenized_text=False):
    """
    加载并缓存参考数据集（Flickr8K）用于蒸馏
    Args:
        return_tokenized_text: 若为 True，DataLoader 每批返回
            (images, input_ids, attention_mask, img_features, txt_features)，
            避免训练循环重复调用 tokenizer；否则返回原始 4 元组。
    Returns: DataLoader or None
    """
    if args.reference_dataset != "flickr8k":
        print("Skipping reference dataset loading.")
        return None
    
    try:
        # 加载 Flickr8k
        from utils_data import Flickr8kDataset
        ref_dataset_obj = Flickr8kDataset(root="/data1/open_datasets/flickr8k/")
        raw_ref_loader = ref_dataset_obj.return_loader(
            batch_size=32, shuffle=False, num_workers=4
        )
        
        # 缓存 Teacher 特征
        cached_imgs, cached_txts = [], []
        cached_t_img_feats, cached_t_txt_feats = [], []
        
        with torch.no_grad():
            for imgs, txts in tqdm(raw_ref_loader, desc="Caching Reference Data"):
                imgs = imgs.to(device)
                t_img_feat = model_pretrain.get_image_features(imgs)
                t_img_feat = t_img_feat / t_img_feat.norm(dim=-1, keepdim=True)
                
                # 编码文本
                text_inputs = processor(text=txts, return_tensors="pt", padding=True, truncation=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                t_txt_feat = model_pretrain.get_text_features(**text_inputs)
                t_txt_feat = t_txt_feat / t_txt_feat.norm(dim=-1, keepdim=True)
                
                cached_imgs.append(imgs.cpu())
                cached_txts.extend(txts)
                cached_t_img_feats.append(t_img_feat.cpu())
                cached_t_txt_feats.append(t_txt_feat.cpu())
        
        # 预 tokenize 所有参考文本，避免训练循环重复 CPU tokenization
        if return_tokenized_text:
            all_texts = cached_txts
            tokenized = processor(text=all_texts, return_tensors="pt", padding=True, truncation=True)
            text_data = (tokenized["input_ids"], tokenized["attention_mask"])
            merged_ref_dataset = MergedReferenceDataset(
                torch.cat(cached_imgs), text_data,
                torch.cat(cached_t_img_feats), torch.cat(cached_t_txt_feats),
                tokenized=True)
        else:
            merged_ref_dataset = MergedReferenceDataset(
                torch.cat(cached_imgs), cached_txts,
                torch.cat(cached_t_img_feats), torch.cat(cached_t_txt_feats),
                tokenized=False)
        reference_loader = DataLoader(
            merged_ref_dataset, batch_size=getattr(args, 'reference_batch_size', 32),
            sampler=InfiniteSampler(merged_ref_dataset, shuffle=True, seed=42),
            num_workers=4, pin_memory=True
        )
        
        print(f"✓ Reference dataset loaded: {len(merged_ref_dataset)} samples")
        return reference_loader
        
    except Exception as e:
        print(f"Warning: Failed to load reference dataset ({e}). Distillation disabled.")
        return None