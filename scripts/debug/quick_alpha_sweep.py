"""快速 alpha 灵敏度分析"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
import os, argparse, logging
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from src.trainers.lora_nsp_trainer import LoRANSPTrainer
from src.classifiers.lr_rgda_classifier import LRRGDAClassifier
from src.classifiers.gaussian_statistics import build_multi_center_stats_dict
from src.utils.main_utils import fix_random_seed, get_zeroshot_classifier
from src.utils.data import get_xtail_trainloader, get_transforms
from src.utils.feature_extractor import extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--num_shots", type=int, default=16)
parser.add_argument("--num_centers", type=int, default=4)
parser.add_argument("--rgda_train_iter", type=int, default=200)
parser.add_argument("--rgda_train_lr", type=float, default=0.01)
parser.add_argument("--gmm_k", type=int, default=4)
args = parser.parse_args()
args.device = f"cuda:{args.gpu}"
args.lora_rank = 4; args.lora_type = 'lora_nsp'
args.tune_vision_encoder = False; args.tune_text_encoder = False
args.nsp_eps = 0.05; args.nsp_weight = 0.02; args.text_lora_rank = 4
args.root = "/data1/open_datasets/X-TAIL"
args.rgda_rank = 32
args.rgda_alpha1 = 0.2; args.rgda_alpha2 = 2.0; args.rgda_alpha3 = 0.5
args.seed = 42

fix_random_seed(args.seed)

# Init
trainer = LoRANSPTrainer(args)
model = trainer.model; processor = trainer.processor

datasets = ['aircraft','caltech101','dtd','eurosat','flowers','food101','mnist','oxford_pets','stanford_cars','sun397']

# Extract training features
logging.info("Extracting training features...")
all_feats, all_raw_feats, all_lbls = [], [], []
feat_offset = 0
all_class_names = []
for d_name in datasets:
    train_transform, _ = get_transforms(d_name)
    tr_loader, _, _, c_names = get_xtail_trainloader(root=args.root, dataset_name=d_name, transform_train=train_transform, transform_test=None, num_shots=args.num_shots, batch_size=32)
    features, labels = extract_features(model, tr_loader, args.device)
    all_raw_feats.append(features.clone())
    features = features / features.norm(dim=-1, keepdim=True)
    all_feats.append(features)
    all_lbls.append(labels + feat_offset)
    feat_offset += len(c_names)
    all_class_names.extend(c_names)

all_features = torch.cat(all_feats)
all_raw_features = torch.cat(all_raw_feats)
all_labels = torch.cat(all_lbls)

# Build stats_dict and LR-RGDA
stats_dict, center_means = build_multi_center_stats_dict(all_features, all_labels, M=args.num_centers)
feat_offset = 0; per_dataset_covs = []
for d_name in datasets:
    _, _, _, c_names = get_xtail_trainloader(root=args.root, dataset_name=d_name, transform_train=None, transform_test=None, num_shots=args.num_shots, batch_size=32)
    n_classes = len(c_names)
    ds_cov = sum(stats_dict[cid].cov for cid in range(feat_offset, feat_offset + n_classes)) / n_classes
    per_dataset_covs.append(ds_cov); feat_offset += n_classes
dataset_balanced_global_cov = sum(per_dataset_covs) / len(per_dataset_covs)

lr_rgda = LRRGDAClassifier(stats_dict=stats_dict, device=args.device, rank=args.rgda_rank, qda_reg_alpha1=args.rgda_alpha1, qda_reg_alpha2=args.rgda_alpha2, qda_reg_alpha3=args.rgda_alpha3, temperature=1.0, M=args.num_centers, center_means=center_means, global_cov=dataset_balanced_global_cov)

# Generate GMM features and fine-tune
logging.info("Generating GMM features...")
from sklearn.mixture import GaussianMixture
all_raw_np = all_raw_features.cpu().numpy()
all_labels_np = all_labels.cpu().numpy()
syn_feats, syn_labels = [], []
for cid in sorted(stats_dict.keys()):
    mask = all_labels_np == cid; feats_c = all_raw_np[mask]
    gmm = GaussianMixture(n_components=args.gmm_k, covariance_type='spherical', random_state=42, reg_covar=1e-6)
    gmm.fit(feats_c)
    n_per_comp = np.maximum(1, (gmm.weights_ * 16).round().astype(int))
    diff = 16 - n_per_comp.sum(); n_per_comp[np.argmax(gmm.weights_)] += diff
    for comp in range(args.gmm_k):
        mean = torch.from_numpy(gmm.means_[comp]).float()
        var = gmm.covariances_[comp]
        n = max(1, int(n_per_comp[comp]))
        noise = torch.randn(n, mean.shape[0]) * np.sqrt(max(var, 1e-8))
        samples = (mean.unsqueeze(0) + noise)
        samples = samples / samples.norm(dim=-1, keepdim=True)
        syn_feats.append(samples.to(args.device))
        syn_labels.append(torch.full((n,), cid, device=args.device))
syn_features = torch.cat(syn_feats); syn_labels_t = torch.cat(syn_labels)

logging.info(f"Fine-tuning LR-RGDA on {syn_features.shape[0]} GMM features...")
lr_rgda.fit(syn_features, syn_labels_t, iterations=args.rgda_train_iter, lr=args.rgda_train_lr)

# ZS classifier
zs_classifier = get_zeroshot_classifier(model, processor, all_class_names, args.device)
num_classes = len(all_class_names)

# Extract test features and alpha sweep
logging.info("Extracting test features for alpha sweep...")
all_test_feats, all_test_lbls = [], []
offset = 0
for d_name in datasets:
    _, test_transform = get_transforms(d_name)
    _, _, te_loader, _ = get_xtail_trainloader(root=args.root, dataset_name=d_name, transform_train=None, transform_test=test_transform, num_shots=args.num_shots, batch_size=32)
    feats, lbls = extract_features(model, te_loader, args.device)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    all_test_feats.append(feats.to(args.device))
    all_test_lbls.append((lbls + offset).to(args.device))
    offset += len(c_names)
test_features = torch.cat(all_test_feats)
test_labels = torch.cat(all_test_lbls)

zs_logits = test_features @ zs_classifier
zs_logits_norm = zs_logits - zs_logits.max(dim=-1, keepdim=True).values
rgda_logits = lr_rgda.forward(test_features)
rgda_logits_norm = rgda_logits - rgda_logits.max(dim=-1, keepdim=True).values

print(f"\n{'Alpha':>8} {'ZS':>8} {'RGDA':>8} {'Ensemble':>10}")
print("-" * 40)
for alpha in np.linspace(0, 1.0, 21):
    ens_logits = zs_logits_norm * (1 - alpha)
    ens_logits[:, :num_classes] += alpha * rgda_logits_norm
    ens_acc = ens_logits.argmax(dim=1).eq(test_labels).float().mean().item() * 100
    zs_acc = zs_logits_norm.argmax(dim=1).eq(test_labels).float().mean().item() * 100
    rgda_acc = rgda_logits_norm.argmax(dim=1).eq(test_labels).float().mean().item() * 100
    print(f"{alpha:8.2f} {zs_acc:7.1f}% {rgda_acc:7.1f}% {ens_acc:9.1f}%")
