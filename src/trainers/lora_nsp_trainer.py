import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.models.clip import get_clip_model
from src.models.utils import feature_distillation_loss, cross_modal_distillation_loss


def symmetric_cross_entropy_loss(logits, targets, sce_a=0.5, sce_b=0.5):
    pred = F.softmax(logits, dim=1)
    pred = torch.clamp(pred, min=1e-7, max=1.0)
    label_one_hot = F.one_hot(targets, pred.size(1)).float().to(pred.device)
    label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
    ce_loss = -torch.sum(label_one_hot * torch.log(pred), dim=1).mean()
    rce_loss = -torch.sum(pred * torch.log(label_one_hot), dim=1).mean()
    return sce_a * ce_loss + sce_b * rce_loss


class FeatureExtractorHook:
    """用于捕获中间层特征的Hook"""
    def __init__(self):
        self.features =[]

    def __call__(self, module, input, output):
        if isinstance(input, tuple):
            x = input[0]
        else:
            x = input
        self.features.append(x.detach().cpu())

    def clear(self):
        self.features =[]

    def get_features(self):
        if len(self.features) == 0:
            return None
        return torch.cat(self.features, dim=0)


class LoRANSPTrainer:
    def __init__(self, args, covariance_history: Optional[Dict[str, torch.Tensor]] = None,
                 text_covariance_history: Optional[Dict[str, torch.Tensor]] = None):
        self.args = args
        self.device = args.device
        self.model, self.processor = get_clip_model(args, train_mode='lora')
        self.model.to(self.device)

        # 检查是否启用了视觉/文本编码器 LoRA
        self.has_vision_lora = hasattr(self.model.vision_model, 'lora_modules')
        self.has_text_lora = hasattr(self.model.text_model, 'lora_modules')
        self.has_text_adapter = self.has_text_lora or hasattr(self.model.text_model, 'adaptformer_modules')

        # 预训练模型（用于蒸馏）
        self.model_pretrain, _ = get_clip_model(args, train_mode="frozen")
        self.model_pretrain.to(self.device)

        # 协方差历史
        self.covariance_history = covariance_history or {}
        self.text_covariance_history = text_covariance_history or {}
        self.cov_momentum = getattr(args, 'cov_momentum', 0.9)
        self.covariance_counts: Dict[str, int] = {}
        self.text_covariance_counts: Dict[str, int] = {}

        # 加载图像协方差历史
        if self.covariance_history and self.has_vision_lora:
            logging.info(f"Loading image covariance history with {len(self.covariance_history)} layers")
            self.model.vision_model.update_projection_matrices(self.covariance_history)
        # 加载文本协方差历史
        if self.text_covariance_history and self.has_text_lora:
            logging.info(f"Loading text covariance history with {len(self.text_covariance_history)} layers")
            self.model.text_model.update_projection_matrices(self.text_covariance_history)

    def encode_text(self, text):
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_outputs = self.model.text_model(**text_inputs)
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            pooled = text_outputs.pooler_output
        elif hasattr(text_outputs, 'last_hidden_state'):
            pooled = text_outputs.last_hidden_state[:, -1, :]
        else:
            pooled = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs
        text_features = self.model.text_projection(pooled)
        return text_features

    def encode_image(self, img):
        return self.model.get_image_features(img)

    def zeroshot_classifier(self, classnames, templates, use_grad=False):
        """构造 ZS 分类器矩阵 [feature_dim, num_classes]（批量编码，可选梯度）"""
        all_texts = []
        class_text_counts = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [template(classname) for template in templates]
            all_texts.extend(texts)
            class_text_counts.append(len(texts))

        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            all_embeddings = self.encode_text(all_texts)
            all_embeddings = all_embeddings / all_embeddings.norm(dim=-1, keepdim=True)

        zeroshot_weights = []
        start = 0
        for count in class_text_counts:
            class_embedding = all_embeddings[start:start + count].mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
            start += count

        return torch.stack(zeroshot_weights, dim=1).to(self.device)

    def get_optimizer(self, params, lr, weight_decay, iterations):
        optimizer = torch.optim.AdamW(params, lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=iterations, eta_min=lr/3)
        return optimizer, scheduler

    def _extract_covariances_from_modules(self, lora_modules, forward_fn, data_iter,
                                           desc="Collecting features"):
        """通用协方差提取"""
        torch.cuda.empty_cache()
        self.model.eval()

        module_names = list(lora_modules.keys())
        logging.info(f"Found {len(module_names)} LoRA modules")

        hooks, feature_extractors, running_xtx = {}, {}, {}
        for name in module_names:
            extractor = FeatureExtractorHook()
            hook = lora_modules[name].register_forward_hook(extractor)
            hooks[name] = hook
            feature_extractors[name] = extractor
            running_xtx[name] = None

        total_observations = 0
        for batch_data in tqdm(data_iter, desc=desc, leave=False):
            batch_input = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data
            batch_input = batch_input.to(self.device)
            _ = forward_fn(batch_input)

            for name in module_names:
                batch_feats = feature_extractors[name].get_features()
                if batch_feats is not None:
                    batch_feats = batch_feats.to(torch.float32)
                    if batch_feats.dim() == 3:
                        batch_feats = batch_feats.reshape(-1, batch_feats.shape[-1])
                    xtx_batch = batch_feats.t() @ batch_feats
                    if running_xtx[name] is None:
                        running_xtx[name] = xtx_batch
                    else:
                        running_xtx[name] += xtx_batch
                    feature_extractors[name].clear()

            if batch_feats is not None:
                total_observations += batch_feats.shape[0]

        covariances = {}
        for name in module_names:
            if running_xtx[name] is not None:
                cov = running_xtx[name] / total_observations
                eps = 1e-6
                cov = (cov + cov.t()) / 2.0
                cov = cov + torch.eye(cov.shape[0], device=cov.device) * eps
                if torch.isnan(cov).any() or torch.isinf(cov).any():
                    logging.warning(f"Layer {name} contains NaN or Inf. Cleaning...")
                    cov = torch.nan_to_num(cov, nan=0.0, posinf=1.0, neginf=-1.0)
                covariances[name] = cov.to('cpu')
            hooks[name].remove()

        logging.info(f"Extracted covariances for {len(covariances)} layers")
        return covariances

    @torch.no_grad()
    def extract_layer_covariances(self, data_loader) -> Dict[str, torch.Tensor]:
        """提取图像编码器 LoRA 层的协方差矩阵"""
        logging.info("=== Extracting Image Encoder Covariances ===")
        return self._extract_covariances_from_modules(
            lora_modules=self.model.vision_model.lora_modules,
            forward_fn=lambda x: self.encode_image(x),
            data_iter=data_loader,
            desc="Collecting image features",
        )

    @torch.no_grad()
    def extract_text_covariances(self, class_names) -> Dict[str, torch.Tensor]:
        """提取文本编码器 LoRA 层的协方差矩阵（用类名模板构造文本）"""
        if not self.has_text_lora:
            logging.info("Text LoRA not enabled, skipping text covariance extraction.")
            return {}

        logging.info("=== Extracting Text Encoder Covariances ===")
        templates = [lambda x: f"a photo of a {x}."]
        all_texts = []
        for classname in class_names:
            classname_clean = classname.replace('_', ' ')
            all_texts.extend([template(classname_clean) for template in templates])

        batch_size = self.args.batch_size
        text_batches = []
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            text_inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
            text_batches.append(text_inputs.input_ids)

        return self._extract_covariances_from_modules(
            lora_modules=self.model.text_model.lora_modules,
            forward_fn=lambda input_ids: self.model.text_model(
                input_ids=input_ids, attention_mask=(input_ids != 0).long()),
            data_iter=text_batches,
            desc="Collecting text features",
        )

    def update_covariance_history(self, new_covariances: Dict[str, torch.Tensor]):
        """更新图像协方差历史并更新投影矩阵（等权平均）"""
        logging.info(f"=== Updating Image Covariance History ===")
        self._apply_covariance_update(
            new_covariances, self.covariance_history, self.covariance_counts,
            self.model.vision_model.update_projection_matrices, "image")

    def update_text_covariance_history(self, new_covariances: Dict[str, torch.Tensor]):
        """更新文本协方差历史并更新投影矩阵（等权平均）"""
        if not new_covariances:
            return
        logging.info(f"=== Updating Text Covariance History ===")
        self._apply_covariance_update(
            new_covariances, self.text_covariance_history, self.text_covariance_counts,
            self.model.text_model.update_projection_matrices, "text")

    def _apply_covariance_update(self, new_covariances, history_dict, count_dict, update_fn, tag):
        """通用协方差等权平均 + 投影矩阵更新"""
        updated, new = 0, 0
        for layer_name, new_cov in new_covariances.items():
            if layer_name in history_dict:
                cnt = count_dict.get(layer_name, 1)
                old_cov = history_dict[layer_name]
                merged_cov = (old_cov * cnt + new_cov) / (cnt + 1)
                history_dict[layer_name] = merged_cov
                count_dict[layer_name] = cnt + 1
                updated += 1
            else:
                history_dict[layer_name] = new_cov
                count_dict[layer_name] = 1
                new += 1
        logging.info(f"  [{tag}] Updated {updated} layers, Added {new} new layers")
        update_fn(history_dict)
        logging.info(f"  [{tag}] Projection matrices updated")
        torch.cuda.empty_cache()

    def finalize_task_for_incremental(self) -> None:
        """增量学习：合并 LoRA 并重置，同时处理图像和文本编码器"""
        logging.info("=== Finalizing Task for Incremental Learning ===")

        # 图像编码器 merge + reset
        if self.has_vision_lora:
            if hasattr(self.model.vision_model, 'merge_and_reset_for_incremental'):
                self.model.vision_model.merge_and_reset_for_incremental()
            else:
                for name, module in self.model.vision_model.lora_modules.items():
                    if hasattr(module, 'merge_lora_weights'):
                        module.merge_lora_weights()
                    elif hasattr(module, 'merge_and_reinit'):
                        module.merge_and_reinit()
                    elif hasattr(module, 'merge_and_reset'):
                        module.merge_and_reset()
                    elif hasattr(module, 'merge'):
                        module.merge()
                    else:
                        raise AttributeError(f"Image LoRA module {name} lacks a merge/reset method.")

        # 文本编码器 merge + reset
        if self.has_text_lora:
            for name, module in self.model.text_model.lora_modules.items():
                if hasattr(module, 'merge_lora_weights'):
                    module.merge_lora_weights()
                elif hasattr(module, 'merge_and_reinit'):
                    module.merge_and_reinit()
                elif hasattr(module, 'merge_and_reset'):
                    module.merge_and_reset()
                elif hasattr(module, 'merge'):
                    module.merge()
                else:
                    raise AttributeError(f"Text LoRA module {name} lacks a merge/reset method.")

        logging.info("Task finalized: LoRA weights merged and reset for enabled encoders.")

    def train(self, train_loader, class_names, reference_loader,
              eval_interval=0, eval_callback=None, aux_weight=0.0,
              train_text_encoder=None, text_lr=None, iterations=None):
        """
        训练模型

        [text LoRA] 若启用 text LoRA：
          - 每步批量编码类名重算 ZS 分类器（use_grad=True），让文本编码器通过 CE 损失获得梯度
          - 增量场景每任务 ~100 类，显存安全；联合场景 1100 类时自动随机采样 max_zs_classes 个类
          - CD 损失使用 student text encoder 编码参考文本
        """
        import random as _random

        templates = [lambda x: f"a photo of a {x}."]
        n_classes = len(class_names)
        max_zs_classes = getattr(self.args, 'max_zs_classes', 128)
        train_iterations = int(iterations if iterations is not None else self.args.iterations)
        if train_iterations <= 0:
            raise ValueError(f"Training iterations must be positive, got {train_iterations}")
        train_text_encoder = self.has_text_adapter if train_text_encoder is None else bool(train_text_encoder)
        text_lr = self.args.lr if text_lr is None else float(text_lr)
        text_grad_enabled = self.has_text_adapter and train_text_encoder and text_lr > 0

        if self.has_text_adapter and text_grad_enabled:
            precomputed_classifier = None
        else:
            precomputed_classifier = self.zeroshot_classifier(class_names, templates)

        # 优化器：vision 和 text LoRA 分组，允许 task-wise text LR/freeze。
        param_groups = []
        if self.has_vision_lora:
            vision_params = list(self.model.vision_model.get_params())
            if vision_params:
                param_groups.append({'params': vision_params, 'lr': self.args.lr})
        if self.has_text_adapter and text_grad_enabled:
            text_params = list(self.model.text_model.get_params())
            if text_params:
                param_groups.append({'params': text_params, 'lr': text_lr})

        base_lr = self.args.lr
        logging.info(
            "Task train schedule: vision_lora=%s, text_adapter=%s, train_text=%s, text_lr=%.6g",
            self.has_vision_lora, self.has_text_adapter, text_grad_enabled, text_lr if text_grad_enabled else 0.0)
        if aux_weight > 0:
            feature_dim = self.model.config.projection_dim
            self.aux_head = nn.Linear(feature_dim, n_classes, bias=False).to(self.device)
            param_groups.append({'params': self.aux_head.parameters(), 'lr': 5e-3})
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        else:
            self.aux_head = None
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        # A scalar eta_min keeps CosineAnnealingLR compatible across PyTorch
        # versions. Use 0 so low text-LR groups are not raised above their
        # initial LR by the previous base_lr / 3 floor.
        scheduler_type = getattr(self.args, "scheduler", "cosine")
        if scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[group["lr"] for group in optimizer.param_groups],
                total_steps=train_iterations,
            )
        elif scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=train_iterations,
                                          eta_min=0.0)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        logit_scale = self.model.logit_scale.detach()

        self.model.train()
        train_iter = iter(train_loader)
        ref_iter = iter(reference_loader) if reference_loader is not None else None

        ema_loss = torch.tensor(0.0)
        ema_acc = torch.tensor(0.0)
        ema_aux_ce = torch.tensor(0.0)
        ema_aux_acc = torch.tensor(0.0)
        ema_fd = torch.tensor(0.0)
        ema_cd = torch.tensor(0.0)

        pbar = tqdm(range(train_iterations), desc="Training")
        for i in pbar:
            try:
                images, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, labels = next(train_iter)

            images = images.to(self.device)
            labels = labels.to(self.device)

            # --- 前向传播 ---
            vision_ctx = torch.no_grad() if not self.has_vision_lora else torch.enable_grad()
            with vision_ctx:
                proj_feats = self.model.get_image_features(images)
            norm_feats = proj_feats / proj_feats.norm(dim=-1, keepdim=True)

            # --- ZS 分类器 ---
            if self.has_text_adapter and text_grad_enabled:
                if n_classes > max_zs_classes:
                    # 联合训练场景：随机采样子集控制显存
                    batch_classes = labels.unique().tolist()
                    n_remaining = max_zs_classes - len(batch_classes)
                    if n_remaining > 0:
                        other_classes = [c for c in range(n_classes) if c not in batch_classes]
                        sampled = _random.sample(other_classes, min(n_remaining, len(other_classes)))
                        zs_class_indices = batch_classes + sampled
                    else:
                        zs_class_indices = batch_classes[:max_zs_classes]
                    zs_class_names = [class_names[idx] for idx in zs_class_indices]
                    classifier = self.zeroshot_classifier(zs_class_names, templates, use_grad=True)
                    idx_to_subidx = {orig: new for new, orig in enumerate(zs_class_indices)}
                    remapped_labels = torch.tensor(
                        [idx_to_subidx.get(lb.item(), -100) for lb in labels], device=labels.device)
                else:
                    # 增量训练场景：全量类名编码（~100 类，显存安全）
                    classifier = self.zeroshot_classifier(class_names, templates, use_grad=True)
                    remapped_labels = labels
            else:
                classifier = precomputed_classifier
                remapped_labels = labels

            logits = logit_scale.exp() * (norm_feats @ classifier)

            # SCE 损失
            ce_loss = symmetric_cross_entropy_loss(
                logits, remapped_labels,
                getattr(self.args, 'sce_a', 0.5), getattr(self.args, 'sce_b', 0.5))
            loss = ce_loss

            preds = logits.argmax(dim=-1)
            valid_mask = (remapped_labels != -100) if self.has_text_adapter and text_grad_enabled and n_classes > max_zs_classes else None
            if valid_mask is not None and valid_mask.any():
                train_acc = (preds[valid_mask] == remapped_labels[valid_mask]).float().mean().item() * 100
            elif valid_mask is not None and not valid_mask.any():
                train_acc = 0.0
            else:
                train_acc = (preds == remapped_labels).float().mean().item() * 100

            aux_ce_val, aux_acc_val = 0.0, 0.0

            # --- 辅助分类头 ---
            if self.aux_head is not None:
                aux_logits = self.aux_head(proj_feats)
                aux_loss = symmetric_cross_entropy_loss(
                    aux_logits, labels,
                    getattr(self.args, 'sce_a', 0.5), getattr(self.args, 'sce_b', 0.5))
                loss = loss + aux_weight * aux_loss
                aux_ce_val = aux_loss.item()
                aux_preds = aux_logits.argmax(dim=-1)
                aux_acc_val = (aux_preds == labels).float().mean().item() * 100

            l_fd_val, l_cd_val = 0.0, 0.0

            # --- 蒸馏损失 ---
            if reference_loader is not None and ref_iter is not None:
                try:
                    r_imgs, r_texts, t_img_f, t_txt_f = next(ref_iter)
                except StopIteration:
                    ref_iter = iter(reference_loader)
                    r_imgs, r_texts, t_img_f, t_txt_f = next(ref_iter)

                r_imgs = r_imgs.to(self.device)
                t_img_f = t_img_f.to(self.device)
                t_txt_f = t_txt_f.to(self.device)

                vision_ctx_ref = torch.no_grad() if not self.has_vision_lora else torch.enable_grad()
                with vision_ctx_ref:
                    s_img_f = self.model.get_image_features(r_imgs)
                s_img_f = s_img_f / s_img_f.norm(dim=-1, keepdim=True)

                l_fd = feature_distillation_loss(t_img_f, s_img_f)
                l_fd_val = l_fd.item()

                # Text adapter enabled: use the adapted text encoder as the
                # semantic anchor, with gradients only when it is trainable.
                if self.has_text_adapter:
                    text_ctx_ref = torch.enable_grad() if text_grad_enabled else torch.no_grad()
                    with text_ctx_ref:
                        s_txt_f = self.encode_text(r_texts)
                        s_txt_f = s_txt_f / s_txt_f.norm(dim=-1, keepdim=True)
                else:
                    s_txt_f = t_txt_f

                l_cd = cross_modal_distillation_loss(
                    logit_scale, s_img_f, s_txt_f, t_img_f, t_txt_f, 2.0)
                l_cd_val = l_cd.item()

                loss = loss + self.args.fd_weight * l_fd + self.args.cd_weight * l_cd

            # --- 反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            ema_loss = 0.95 * ema_loss + 0.05 * loss.item()
            ema_acc = 0.95 * ema_acc + 0.05 * train_acc
            ema_aux_ce = 0.95 * ema_aux_ce + 0.05 * aux_ce_val
            ema_aux_acc = 0.95 * ema_aux_acc + 0.05 * aux_acc_val
            ema_fd = 0.95 * ema_fd + 0.05 * l_fd_val
            ema_cd = 0.95 * ema_cd + 0.05 * l_cd_val

            pbar.set_postfix({
                'Loss': f"{ema_loss.item():.3f}",
                'Acc': f"{ema_acc.item():.1f}%",
                'AuxCE': f"{ema_aux_ce.item():.3f}",
                'AuxAcc': f"{ema_aux_acc.item():.1f}%",
                'FD': f"{ema_fd.item():.4f}",
                'CD': f"{ema_cd.item():.4f}",
            })

            if (i + 1) % 50 == 0 or (i + 1) == train_iterations:
                logging.info(f"Iter[{i+1:03d}/{train_iterations}] | "
                             f"Loss: {ema_loss.item():.4f} | Acc: {ema_acc.item():.2f}% | "
                             f"AuxCE: {ema_aux_ce.item():.4f} | AuxAcc: {ema_aux_acc.item():.2f}% | "
                             f"FD: {ema_fd.item():.4f} | CD: {ema_cd.item():.4f}")

            if eval_interval > 0 and eval_callback is not None and (i + 1) % eval_interval == 0:
                eval_callback(self.model, i + 1)
                self.model.train()

        return self.model

    def evaluate(self, test_loader, class_names):
        self.model.eval()
        correct = 0
        total = 0

        templates = [lambda x: f"a photo of a {x}."]
        classifier = self.zeroshot_classifier(class_names, templates)
        logit_scale = self.model.logit_scale.detach()

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                img_feats = self.encode_image(images)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

                logits = logit_scale.exp() * (img_feats @ classifier)
                _, predicted = torch.max(logits, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def save_checkpoint(self, path, class_names, stats_dict=None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'covariance_history': self.covariance_history,
            'text_covariance_history': self.text_covariance_history,
            'cov_momentum': self.cov_momentum,
            'class_names': class_names,
            'stats_dict': stats_dict,
            'args': vars(self.args),
        }
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved: {path}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path, args, device='cuda'):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        covariance_history = checkpoint.get('covariance_history', {})
        text_covariance_history = checkpoint.get('text_covariance_history', {})

        trainer = cls(args, covariance_history=covariance_history,
                      text_covariance_history=text_covariance_history)
        trainer.cov_momentum = checkpoint.get('cov_momentum', 0.9)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        return trainer
