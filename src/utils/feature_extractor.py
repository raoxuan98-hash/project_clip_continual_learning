import torch
from tqdm import tqdm
from src.models.backbone_utils import encode_image_features

@torch.inference_mode()
def extract_features(model, dataloader, device, normalize=True, keep_on_device=False):
    """
    从数据加载器中提取特征
    Args:
        model: CLIP模型
        dataloader: 数据加载器
        device: 设备
        normalize: 是否对特征做 L2 归一化
        keep_on_device: 是否将特征保留在 GPU 上（避免 CPU↔GPU 往返，
                        仅在测试集规模可控时使用）
    Returns:
        features: 提取的特征
        labels: 对应的标签
    """
    model.eval()
    features = []
    labels = []

    for images, lbls in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        feats = encode_image_features(model, images)
        if normalize:
            feats = torch.nn.functional.normalize(feats, dim=-1)
        features.append(feats if keep_on_device else feats.cpu())
        labels.append(lbls if keep_on_device else lbls.cpu())

    features = torch.cat(features)
    labels = torch.cat(labels)
    return features, labels

@torch.inference_mode()
def extract_features_for_datasets(model, dataset_names, args, transform, device):
    """
    提取多个数据集的特征
    Args:
        model: CLIP模型
        dataset_names: 数据集名称列表
        args: 配置参数
        transform: 数据变换
        device: 设备
    Returns:
        features_dict: 数据集特征字典
    """
    from src.utils.data import get_xtail_trainloader
    
    features_dict = {}
    for d_name in dataset_names:
        tr_loader, _, _, _ = get_xtail_trainloader(
            root=args.root, dataset_name=d_name,
            transform_train=transform, transform_test=None,
            num_shots=args.num_shots, batch_size=32
        )
        feats, _ = extract_features(model, tr_loader, device)
        features_dict[d_name] = feats
    return features_dict
