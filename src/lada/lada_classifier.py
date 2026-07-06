import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LADAClassifier(nn.Module):
    """
    LADA 分类器：基于 k-means 聚类中心 + 指数亲和变换

    核心公式：
        affinity = image_features @ lada_features  # (N, K_total)
        lada_logits = exp(-β * (1 - affinity)) @ classifier  # (N, C_total)

    增量学习：
        - prev_lada_features: (D, K_prev) 历史任务的聚类中心（冻结）
        - curr_lada_features: (D, K_curr) 当前任务的聚类中心（可学习）
        - joint_classifier: (K_total, C_total) 块对角 one-hot 矩阵
    """

    VALID_SCORE_MODES = ("exp_sum", "linear_sum", "linear_max")

    def __init__(self, feature_dim: int, beta: float = 1.0, score_mode: str = "exp_sum"):
        super().__init__()
        self.feature_dim = feature_dim
        self.beta = beta
        self.score_mode = score_mode
        if score_mode not in self.VALID_SCORE_MODES:
            raise ValueError(
                f"Unsupported LADA score_mode={score_mode!r}. "
                f"Expected one of: {', '.join(self.VALID_SCORE_MODES)}."
            )

        self.register_buffer('prev_lada_features', torch.empty(feature_dim, 0))
        self.register_buffer('joint_classifier', torch.empty(0, 0))

        self.curr_lada_features = None
        self.curr_classifier = None

        self.num_prev_classes = 0
        self.num_curr_classes = 0

    def set_score_mode(self, score_mode: str):
        if score_mode not in self.VALID_SCORE_MODES:
            raise ValueError(
                f"Unsupported LADA score_mode={score_mode!r}. "
                f"Expected one of: {', '.join(self.VALID_SCORE_MODES)}."
            )
        self.score_mode = score_mode

    def build_from_data(self, features, labels, k=16, label_offset=0):
        """
        从训练数据构建当前任务的 LADA features

        Args:
            features: (N, D) 归一化后的图像特征
            labels: (N,) 类别标签（局部空间，从 0 开始）
            k: 每类聚类中心数
            label_offset: 当前任务在全局标签空间的偏移量
        """
        device = features.device
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()

        unique_labels = np.unique(labels_np)
        num_classes = len(unique_labels)

        selected_features = []
        selected_labels = []

        for lbl in unique_labels:
            lbl_indices = np.where(labels_np == lbl)[0]
            lbl_features = features_np[lbl_indices]
            unique_features = np.unique(lbl_features, axis=0)

            actual_k = min(k, len(unique_features))
            if actual_k == len(unique_features):
                cluster_centers = unique_features
            else:
                kmeans = KMeans(n_clusters=actual_k, n_init=10, random_state=42).fit(lbl_features)
                cluster_centers = kmeans.cluster_centers_

            selected_features.append(cluster_centers)
            selected_labels.append(np.full(actual_k, lbl, dtype=labels_np.dtype))

        selected_features = np.concatenate(selected_features, axis=0)
        selected_labels = np.concatenate(selected_labels, axis=0)

        selected_features = torch.from_numpy(selected_features).float().to(device)
        selected_labels = torch.from_numpy(selected_labels).long().to(device)

        lada_features = selected_features.t()
        self.curr_lada_features = nn.Parameter(lada_features)

        curr_classifier = F.one_hot(selected_labels, num_classes=num_classes).float()
        self.curr_classifier = curr_classifier

        N1, D1 = self.joint_classifier.shape if self.joint_classifier.numel() > 0 else (0, 0)
        N2, D2 = curr_classifier.shape
        joint_classifier = torch.zeros(N1 + N2, D1 + D2, device=device)
        if N1 > 0 and D1 > 0:
            joint_classifier[:N1, :D1] = self.joint_classifier
        joint_classifier[N1:, D1:] = curr_classifier
        self.register_buffer('joint_classifier', joint_classifier)

        self.num_curr_classes = num_classes

        logging.info(f"LADA built: {selected_features.shape[0]} cluster centers "
                     f"for {num_classes} classes (k={k})")

    def forward(self, image_features, lada_features=None):
        """
        计算 LADA logits

        Args:
            image_features: (N, D) 归一化后的图像特征
            lada_features: 可选，覆盖默认的 prev+curr 拼接

        Returns:
            lada_logits: (N, C_total)
        """
        if lada_features is None:
            if self.curr_lada_features is not None:
                lada_features = torch.cat([self.prev_lada_features,
                                            self.curr_lada_features], dim=1)
            else:
                lada_features = self.prev_lada_features

        device = image_features.device
        lada_features = lada_features.to(device)
        affinity = image_features @ lada_features
        joint_classifier = self.joint_classifier.to(device)

        if self.score_mode == "exp_sum":
            return torch.exp(-self.beta * (1 - affinity)) @ joint_classifier
        if self.score_mode == "linear_sum":
            return affinity @ joint_classifier
        if self.score_mode == "linear_max":
            class_masks = joint_classifier.t().bool()
            per_class_logits = []
            for class_mask in class_masks:
                if class_mask.any():
                    per_class_logits.append(
                        affinity[:, class_mask].max(dim=1).values
                    )
                else:
                    per_class_logits.append(
                        torch.full((affinity.shape[0],), -float("inf"), device=device)
                    )
            return torch.stack(per_class_logits, dim=1)
        raise RuntimeError(f"Unexpected LADA score_mode={self.score_mode!r}")

    def fit(self, features, labels, iterations=100, lr=0.01, verbose=True):
        """
        用梯度下降微调 curr_lada_features

        Args:
            features: (N, D) 归一化后的图像特征
            labels: (N,) 全局标签空间中的类别标签
            iterations: 训练迭代次数
            lr: 学习率
            verbose: 是否打印训练日志
        """
        if self.curr_lada_features is None:
            logging.warning("LADA fit: no curr_lada_features to train")
            return

        device = features.device
        features = features.to(device)
        labels = labels.to(device)

        optimizer = torch.optim.AdamW([self.curr_lada_features], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iterations, eta_min=lr / 10)

        for i in range(iterations):
            logits = self.forward(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if verbose and (i == 0 or (i + 1) % max(1, iterations // 5) == 0):
                acc = logits.argmax(dim=-1).eq(labels).float().mean().item() * 100
                logging.info(f"LADA-fit [{i+1}/{iterations}] "
                             f"loss={loss.item():.4f} acc={acc:.1f}%")

    def finalize_task(self):
        """
        任务结束后：
        1. 将 curr_lada_features 冻结并追加到 prev_lada_features
        2. curr_lada_features 置空
        """
        if self.curr_lada_features is not None:
            self.register_buffer('prev_lada_features', torch.cat([
                self.prev_lada_features,
                self.curr_lada_features.detach()
            ], dim=1))
            self.curr_lada_features = None
            self.curr_classifier = None
            self.num_prev_classes += self.num_curr_classes
            self.num_curr_classes = 0
            logging.info(f"LADA finalized: prev_lada_features shape = {self.prev_lada_features.shape}")

    def get_all_lada_features(self):
        """返回拼接后的所有 LADA 特征 (D, K_total)"""
        if self.curr_lada_features is not None:
            return torch.cat([self.prev_lada_features, self.curr_lada_features], dim=1)
        return self.prev_lada_features

    def get_total_classes(self):
        """返回当前覆盖的总类别数"""
        return self.joint_classifier.shape[1] if self.joint_classifier.numel() > 0 else 0
