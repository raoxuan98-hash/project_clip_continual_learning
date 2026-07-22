import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, Optional, List
import logging

from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.lada.lada_classifier import LADAClassifier
from src.lada.dpt import DPTManager
from src.utils.feature_extractor import extract_features
from src.models.backbone_utils import embedding_dim, encode_image_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def weighted_cross_entropy(logits, targets, weights=None):
    loss_per_sample = F.cross_entropy(logits, targets, reduction='none')
    if weights is not None:
        return (loss_per_sample * weights).mean()
    return loss_per_sample.mean()


class LADATrainer(LoRANSPTrainer):
    """
    LADA 训练器：继承 LoRANSPTrainer，添加 LADA 分类器和 DPT 训练逻辑

    关键改动：
    1. 训练前：从当前任务数据构建 curr_lada_features（k-means）
    2. 训练中：将 curr_lada_features 加入 optimizer，计算 LADA logits
    3. 训练后：finalize LADA，更新 DPT 原型

    训练循环（无 FD/CD）：
        total_logits = text_logits + α * lada_logits
        loss = weighted_CE(total_logits, all_labels, weights)
    """

    def __init__(self, args, covariance_history=None, text_covariance_history=None,
                 lada_state=None, dpt_state=None):
        super().__init__(args, covariance_history, text_covariance_history)

        feature_dim = embedding_dim(self.model)

        self.lada_classifier = LADAClassifier(
            feature_dim,
            beta=getattr(args, 'lada_beta', 1.0)
        ).to(self.device)

        self.dpt = DPTManager(
            feature_dim,
            prototype_k=getattr(args, 'prototype_k', 4)
        )

        if lada_state is not None:
            self.lada_classifier.load_state_dict(lada_state['classifier'])
            self.lada_classifier.prev_lada_features = lada_state['prev_features'].to(self.device)
            self.lada_classifier.joint_classifier = lada_state['joint_classifier'].to(self.device)
            self.lada_classifier.num_prev_classes = lada_state['num_prev_classes']

        if dpt_state is not None:
            self.dpt.image_prototypes = dpt_state['image_prototypes'].to(self.device)
            self.dpt.image_prototypes_covs = dpt_state['image_prototypes_covs'].to(self.device)
            self.dpt.image_prototypes_weights = dpt_state['image_prototypes_weights'].to(self.device)
            self.dpt.prototype_labels = dpt_state['prototype_labels'].to(self.device)
            self.dpt.text_prototypes = dpt_state['text_prototypes'].to(self.device)

        self.lada_alpha = getattr(args, 'lada_alpha', 1.0)
        self.lada_k = getattr(args, 'lada_k', 16)
        self.replay_mode = getattr(
            args, 'lada_replay_mode',
            'dpt' if getattr(args, 'enable_dpt', False) else 'none'
        )
        self.enable_dpt = self.replay_mode != 'none'
        self.official_mode = getattr(args, 'lada_official_mode', False)
        self.dpt_feature_normalize = getattr(args, 'dpt_feature_normalize', False)
        if self.official_mode and self.dpt_feature_normalize:
            logging.warning(
                "Official LADA fits DPT GMMs in unnormalized CLIP embedding space; "
                "overriding dpt_feature_normalize=True to False.")
            self.dpt_feature_normalize = False
        self.image_prototypes_weight_coef = getattr(args, 'image_prototypes_weight_coef', 64.0)
        logging.info(
            "LADA replay configuration: mode=%s, official_mode=%s, normalize_gmm_features=%s",
            self.replay_mode, self.official_mode, self.dpt_feature_normalize)

    @torch.no_grad()
    def _extract_features_manual(self, dataloader, normalize=True):
        self.model.eval()
        all_features = []
        all_labels = []
        for images, lbls in tqdm(dataloader, desc="Extracting features"):
            images = images.to(self.device)
            feats = encode_image_features(self.model, images)
            if normalize:
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu())
            all_labels.append(lbls)
        return torch.cat(all_features), torch.cat(all_labels)

    def build_lada_from_loader(self, prototype_loader, label_offset=0):
        """
        从训练数据加载器构建 LADA features

        1. 提取特征
        2. k-means 聚类
        3. 构建 curr_lada_features 和 joint_classifier
        """
        logging.info("=== Building LADA features from training data ===")
        features, labels = self._extract_features_manual(prototype_loader, normalize=True)
        features = features.to(self.device)
        labels = labels.to(self.device)

        self.lada_classifier.build_from_data(features, labels, k=self.lada_k,
                                             label_offset=label_offset)
        return features, labels

    def train(self, train_loader, class_names, reference_loader=None,
              aux_weight=0.0, label_offset=0, all_class_names=None,
              prototype_loader=None):
        """
        LADA 训练循环

        Args:
            train_loader: 当前任务的训练数据加载器
            class_names: 当前任务的类名列表
            reference_loader: 参考数据集（不使用，保持接口兼容）
            aux_weight: 辅助分类头权重（不使用，保持接口兼容）
            label_offset: 当前任务在全局标签空间的偏移量
            all_class_names: 所有已见类名列表（含当前任务）
            prototype_loader: 用确定性变换提取 LADA 初始化特征的 loader
        """
        import random as _random

        n_classes = len(class_names)
        c_prev = label_offset
        c_total = c_prev + n_classes

        templates = [lambda x: f"a photo of a {x}."]
        max_zs_classes = getattr(self.args, 'max_zs_classes', 128)

        if prototype_loader is None:
            prototype_loader = train_loader
        features, labels = self.build_lada_from_loader(prototype_loader, label_offset=0)

        if not self.has_text_lora:
            if all_class_names is not None:
                text_classifier = self.zeroshot_classifier(all_class_names, templates)
            else:
                text_classifier = self.zeroshot_classifier(class_names, templates)
        else:
            text_classifier = None

        trainable_params = []
        if self.has_vision_lora:
            trainable_params += list(self.model.vision_model.get_params())
        if self.has_text_lora:
            trainable_params += list(self.model.text_model.get_params())

        param_groups = [{'params': trainable_params, 'lr': self.args.lr}]
        if self.lada_classifier.curr_lada_features is not None:
            param_groups.append({
                'params': [self.lada_classifier.curr_lada_features],
                'lr': self.args.lr
            })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.iterations,
                                      eta_min=self.args.lr / 3)

        logit_scale = self.model.logit_scale.detach()

        self.model.train()
        self.lada_classifier.train()
        train_iter = iter(train_loader)

        ema_loss = torch.tensor(0.0)
        ema_acc = torch.tensor(0.0)

        pbar = tqdm(range(self.args.iterations), desc="Training (LADA)")
        for i in pbar:
            try:
                images, batch_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, batch_labels = next(train_iter)

            images = images.to(self.device)
            batch_labels = batch_labels.to(self.device)

            vision_ctx = torch.no_grad() if not self.has_vision_lora else torch.enable_grad()
            with vision_ctx:
                raw_img_feats = encode_image_features(self.model, images)

            shifted_labels = batch_labels + c_prev

            if self.enable_dpt and self.dpt.has_prototypes():
                phantom_feats, phantom_labels, phantom_weights = self.dpt.sample_prototypes(
                    self.device, add_noise=(self.replay_mode == 'dpt'))
                # DPT replay and current features share the raw CLIP embedding
                # space. Normalize them together before text/LADA classification.
                all_feats = torch.cat([phantom_feats.detach(), raw_img_feats], dim=0)
                all_labels = torch.cat([phantom_labels, shifted_labels], dim=0)
                real_weights = torch.ones(images.shape[0], device=self.device)
                all_weights = torch.cat([
                    phantom_weights * self.image_prototypes_weight_coef,
                    real_weights
                ], dim=0)
            else:
                all_feats = raw_img_feats
                all_labels = shifted_labels
                all_weights = None
            all_feats = F.normalize(all_feats, dim=-1)

            if self.has_text_lora:
                if n_classes > max_zs_classes:
                    batch_classes = batch_labels.unique().tolist()
                    n_remaining = max_zs_classes - len(batch_classes)
                    if n_remaining > 0:
                        other_classes = [c for c in range(n_classes) if c not in batch_classes]
                        sampled = _random.sample(other_classes, min(n_remaining, len(other_classes)))
                        zs_class_indices = batch_classes + sampled
                    else:
                        zs_class_indices = batch_classes[:max_zs_classes]
                    zs_class_names = [class_names[idx] for idx in zs_class_indices]
                    live_text_feats = self.zeroshot_classifier(zs_class_names, templates,
                                                               use_grad=True)

                    if self.dpt.get_num_text_prototypes() > 0:
                        text_protos = self.dpt.text_prototypes.detach().t()
                        full_text_classifier = torch.cat([text_protos, live_text_feats], dim=1)
                    else:
                        full_text_classifier = live_text_feats

                    idx_to_subidx = {orig: new for new, orig in enumerate(zs_class_indices)}
                    remapped_batch_labels = torch.tensor(
                        [idx_to_subidx.get(lb.item(), -100) for lb in batch_labels],
                        device=self.device)
                    remapped_shifted_labels = remapped_batch_labels + c_prev

                    if self.enable_dpt and self.dpt.has_prototypes():
                        all_labels_for_text = torch.cat([phantom_labels, remapped_shifted_labels])
                    else:
                        all_labels_for_text = remapped_shifted_labels
                else:
                    live_text_feats = self.zeroshot_classifier(class_names, templates,
                                                               use_grad=True)
                    if self.dpt.get_num_text_prototypes() > 0:
                        text_protos = self.dpt.text_prototypes.detach().t()
                        full_text_classifier = torch.cat([text_protos, live_text_feats], dim=1)
                    else:
                        full_text_classifier = live_text_feats
                    all_labels_for_text = all_labels

                text_logits = logit_scale.exp() * (all_feats @ full_text_classifier)
            else:
                text_logits = logit_scale.exp() * (all_feats @ text_classifier)
                all_labels_for_text = all_labels

            lada_logits = self.lada_classifier(all_feats)

            if self.official_mode:
                total_logits = text_logits + self.lada_alpha * lada_logits
            else:
                text_preds = text_logits.argmax(dim=1)
                lada_classes = lada_logits.shape[1]
                mask = (text_preds < lada_classes).float().unsqueeze(1)
                total_logits = text_logits + mask * self.lada_alpha * lada_logits

            loss = weighted_cross_entropy(total_logits, all_labels_for_text, all_weights)

            preds = total_logits.argmax(dim=-1)
            if self.enable_dpt and self.dpt.has_prototypes():
                real_start = phantom_feats.shape[0]
                real_preds = preds[real_start:]
                train_acc = real_preds.eq(shifted_labels).float().mean().item() * 100
            else:
                train_acc = preds.eq(all_labels_for_text).float().mean().item() * 100

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            ema_loss = 0.95 * ema_loss + 0.05 * loss.item()
            ema_acc = 0.95 * ema_acc + 0.05 * train_acc

            pbar.set_postfix({
                'Loss': f"{ema_loss.item():.3f}",
                'Acc': f"{ema_acc.item():.1f}%",
            })

            if (i + 1) % 50 == 0 or (i + 1) == self.args.iterations:
                logging.info(f"Iter[{i+1:03d}/{self.args.iterations}] | "
                             f"Loss: {ema_loss.item():.4f} | Acc: {ema_acc.item():.2f}%")

        self.model.eval()
        self.lada_classifier.eval()
        return self.model

    def finalize_lada_task(self, prototype_loader, class_names, label_offset):
        """
        任务结束处理：
        1. 冻结 LADA 特征
        2. 更新 DPT 图像原型
        3. 更新 DPT 文本原型
        """
        logging.info("=== Finalizing LADA Task ===")

        self.lada_classifier.finalize_task()

        if self.enable_dpt:
            features, labels = self._extract_features_manual(
                prototype_loader, normalize=self.dpt_feature_normalize)
            features = features.to(self.device)
            labels = labels.to(self.device)
            self.dpt.update_image_prototypes(features, labels, label_offset,
                                             k=self.dpt.prototype_k)

        templates = [lambda x: f"a photo of a {x}."]
        text_feats = self.zeroshot_classifier(class_names, templates)
        text_feats = text_feats.t()
        self.dpt.update_text_prototypes(text_feats)

    def get_lada_state(self):
        """保存 LADA 状态"""
        return {
            'classifier': self.lada_classifier.state_dict(),
            'prev_features': self.lada_classifier.prev_lada_features.cpu(),
            'joint_classifier': self.lada_classifier.joint_classifier.cpu(),
            'num_prev_classes': self.lada_classifier.num_prev_classes,
        }

    def get_dpt_state(self):
        """保存 DPT 状态"""
        return {
            'image_prototypes': self.dpt.image_prototypes.cpu(),
            'image_prototypes_covs': self.dpt.image_prototypes_covs.cpu(),
            'image_prototypes_weights': self.dpt.image_prototypes_weights.cpu(),
            'prototype_labels': self.dpt.prototype_labels.cpu(),
            'text_prototypes': self.dpt.text_prototypes.cpu(),
        }
