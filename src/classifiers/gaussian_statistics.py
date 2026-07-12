from __future__ import annotations

import torch
from typing import Dict, Optional
from tqdm import tqdm


def kmeans(x: torch.Tensor, M: int, n_iter: int = 20, seed: int = 42) -> torch.Tensor:
    """
    类内 k-means 聚类，返回 M 个中心。

    Args:
        x: [N, D]，一个类的所有特征
        M: 中心数
        n_iter: 最大迭代次数
        seed: 随机种子

    Returns:
        centers: [M, D]
    """
    unique_x = torch.unique(x, dim=0)
    if unique_x.size(0) <= M:
        # Replay modes such as GMM-mean can contain repeated pseudo-features.
        # Preserve the requested fixed center count for downstream tensors by
        # cycling unique centers when there are fewer unique points than M.
        if unique_x.size(0) == M:
            return unique_x
        repeat = (M + unique_x.size(0) - 1) // unique_x.size(0)
        return unique_x.repeat((repeat, 1))[:M]

    N, D = x.shape
    # 使用局部 RNG 避免全局种子污染
    rng = torch.Generator(device=x.device).manual_seed(seed + hash(str(x.shape)) % 2**31)
    idx = torch.randperm(N, generator=rng)[:M]
    centers = x[idx].clone()

    for _ in range(n_iter):
        # 分配：计算每个点到所有中心的距离
        dists = torch.cdist(x, centers)  # [N, M]
        labels = dists.argmin(dim=1)  # [N]

        # 更新中心
        new_centers = torch.zeros_like(centers)
        for j in range(M):
            mask = labels == j
            if mask.sum() > 0:
                new_centers[j] = x[mask].mean(dim=0)
            else:
                # 空簇：保留原中心
                new_centers[j] = centers[j]

        if torch.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return centers


def build_multi_center_stats_dict(
    features: torch.Tensor,
    labels: torch.Tensor,
    M: int = 1,
    kmeans_seed: int = 42
) -> Dict[int, GaussianStatistics]:
    """
    从特征和标签构建单中心或多中心统计分布。

    M=1: 返回标准 Dict[class_id, GaussianStatistics(mean, cov)]
    M>1: 每个类通过 k-means 得到 M 个中心，共享类协方差。
         返回的 GaussianStatistics.mean 存储展平后的中心拼接 [M*D]，
         需配合 center_means 字典使用。

    Args:
        features: [N, D]
        labels: [N]
        M: 每类的中心数（1=单中心）
        kmeans_seed: k-means 随机种子

    Returns:
        stats_dict: Dict[class_id, GaussianStatistics]
        center_means: Dict[class_id, Tensor[M, D]]（仅 M>1 时）
    """
    unique_classes = torch.unique(labels)
    stats_dict = {}
    center_dict = {}

    for c in unique_classes:
        idx = (labels == c)
        data_c = features[idx]

        # 协方差（始终共享）
        if data_c.shape[0] > 1:
            cov = torch.cov(data_c.T)
        else:
            cov = torch.eye(data_c.shape[1], device=data_c.device)

        if M == 1:
            # 单中心：标准均值
            mu = torch.mean(data_c, dim=0)
            stats_dict[int(c)] = GaussianStatistics(mu, cov)
        else:
            # 多中心：k-means，用 class_id 做种子偏移确保每类不同
            centers = kmeans(data_c, M, seed=kmeans_seed + int(c))
            # 主均值（兼容旧接口，取第一个中心）
            mu = centers[0]
            stats_dict[int(c)] = GaussianStatistics(mu, cov)
            center_dict[int(c)] = centers

    if M > 1:
        return stats_dict, center_dict
    return stats_dict, None


def cholesky_stable(matrix: torch.Tensor, reg: float = 1e-5) -> torch.Tensor:
    if matrix.dim() == 3:
        batch_size, n, _ = matrix.shape
        reg_eye = reg * torch.eye(n, device=matrix.device, dtype=matrix.dtype)
        reg_eye = reg_eye.unsqueeze(0).repeat(batch_size, 1, 1)
        return torch.linalg.cholesky(matrix + reg_eye)
    
    elif matrix.dim() == 2:
        reg_eye = reg * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
        return torch.linalg.cholesky(matrix + reg_eye)
    else:
        raise ValueError(f"不支持的矩阵维度: {matrix.dim()}。支持2D或3D张量。")


class GaussianStatistics:
    """Container for per-class Gaussian statistics."""
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor, reg: float = 1e-4, cholesky = False):
        if mean.dim() == 2 and mean.size(0) == 1:
            mean = mean.squeeze(0)
        if mean.dim() != 1:
            raise AssertionError("GaussianStatistics.mean 必须是 1D 向量")

        self.mean = mean
        self.cov = cov
        self.reg = reg

        if cholesky:
            self.L = cholesky_stable(cov, reg=reg)
        else:
            self.L = None

    def to(self, device):
        """Move statistics to the requested device."""

        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        if self.L is not None:
            self.L = self.L.to(device)
        return self

    def sample(
        self,
        n_samples = None,
        cached_eps = None,
    ) -> torch.Tensor:
        """Draw samples from the Gaussian distribution."""

        if self.L is None:
            self.L = cholesky_stable(self.cov, reg=self.reg)

        device = self.mean.device
        d = self.mean.size(0)

        if cached_eps is None:
            if n_samples is None:
                raise ValueError("n_samples 必须在未提供 cached_eps 时给定")
            eps = torch.randn(n_samples, d, device=device)
        else:
            eps = cached_eps.to(device)
            n_samples = eps.size(0)

        samples = self.mean.unsqueeze(0) + eps @ self.L.t()
        return samples

class LowRankGaussianStatistics:
    def __init__(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        rank: int = 512,
        reg: float = 1e-8, 
        device=None
    ):
        if mean.dim() != 1:
            raise ValueError("mean must be a 1D vector")
        
        d = mean.size(0)
        if cov.shape != (d, d):
            raise ValueError("cov shape mismatch")

        self.mean = mean
        self.d = d
        self.rank = rank or min(100, d)  # default max rank=100
        self.reg = reg

        U, S, Vh = torch.svd_lowrank(cov, q=rank, niter=4, M=None)
        U = U[:, :self.rank]
        S = S[:self.rank]

        self.U = U
        self.S = torch.clamp(S, min=0.0)

        if device:
            self.to(device)
    @property
    def L(self):
        return self.U * torch.sqrt(self.S).unsqueeze(0)

    @property
    def cov(self) -> torch.Tensor:
        """Reconstruct full covariance (use sparingly for high d!)"""
        return self.L @ self.L.T

    def to(self, device):
        self.mean = self.mean.to(device)
        return self

    def sample(self, n_samples: int) -> torch.Tensor:
        """Efficient sampling without forming full covariance"""
        eps = torch.randn(n_samples, self.L.size(1), device=self.L.device, dtype=self.L.dtype)
        return self.mean.unsqueeze(0) + eps @ self.L.T  # (n_samples, d)


def build_stats_dict_from_features(
    features: torch.Tensor,
    labels: torch.Tensor
) -> Dict[int, 'GaussianStatistics']:
    """
    从特征和标签构建类别统计分布字典

    Args:
        features: 特征向量 [N, D]
        labels: 标签 [N]

    Returns:
        类别统计分布字典 {class_id: GaussianStatistics}
    """
    unique_classes = torch.unique(labels)
    stats_dict = {}

    for c in unique_classes:
        idx = (labels == c)
        data_c = features[idx]
        mu = torch.mean(data_c, dim=0)
        if data_c.shape[0] > 1:
            cov = torch.cov(data_c.T)
        else:
            cov = torch.eye(data_c.shape[1], device=data_c.device)

        stats_dict[int(c)] = GaussianStatistics(mu, cov)

    return stats_dict


def extract_stats_dict_from_model(
    model,
    dataset_names: list,
    args,
    device: str = "cuda"
) -> Dict[int, 'GaussianStatistics']:
    """
    从数据集和模型提取类别统计分布

    Args:
        model: CLIP模型或其他特征提取器
        dataset_names: 数据集名称列表
        args: 参数对象，需包含root, num_shots等
        device: 计算设备

    Returns:
        类别统计分布字典 {class_id: GaussianStatistics}
    """
    from src.utils.data import get_xtail_trainloader, get_transforms

    model.eval()
    stats_dict = {}
    label_offset = 0

    for d_name in dataset_names:
        transform, _ = get_transforms(d_name)
        tr_loader, _, _, c_names = get_xtail_trainloader(
            root=args.root,
            dataset_name=d_name,
            transform_train=transform,
            transform_test=None,
            num_shots=args.num_shots,
            batch_size=32
        )

        all_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, lbls in tqdm(tr_loader, desc=f"Extracting {d_name}", leave=False):
                imgs = imgs.to(device)
                feats = model.get_image_features(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu())
                all_labels.append(lbls + label_offset)

        all_feats = torch.cat(all_feats)
        all_labels = torch.cat(all_labels)

        unique_labels = torch.unique(all_labels)
        for c in unique_labels:
            idx = (all_labels == c)
            data_c = all_feats[idx]
            mu = torch.mean(data_c, dim=0)
            if data_c.shape[0] > 1:
                cov = torch.cov(data_c.T)
            else:
                cov = torch.eye(data_c.shape[1])

            stats_dict[int(c)] = GaussianStatistics(mu, cov)

        label_offset += len(c_names)

    return stats_dict
