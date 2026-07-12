import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import os, argparse
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from src.models.clip import get_clip_model
from src.utils.data import get_xtail_trainloader, get_transforms
from src.utils.feature_extractor import extract_features

args = argparse.Namespace(lora_rank=4, lora_type='lora_nsp',
                          tune_vision_encoder=False, tune_text_encoder=False,
                          nsp_eps=0.05, nsp_weight=0.02, text_lora_rank=4)
model, processor = get_clip_model(args, train_mode='lora')
model.eval()
model.cuda()

root = '/data1/open_datasets/X-TAIL'
datasets = ['aircraft', 'caltech101', 'flowers', 'food101']

all_vars = []

for d_name in datasets:
    train_transform, _ = get_transforms(d_name)
    tr_loader, _, _, c_names = get_xtail_trainloader(
        root=root, dataset_name=d_name, transform_train=train_transform,
        transform_test=None, num_shots=16, batch_size=32)
    features, labels = extract_features(model, tr_loader, 'cuda')
    features = features / features.norm(dim=-1, keepdim=True)
    feats_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    print(f"\n=== {d_name} ({len(np.unique(labels_np))} classes) ===")
    vars_for_dataset = []
    for cid in np.unique(labels_np)[:5]:
        mask = labels_np == cid
        feats_c = feats_np[mask]

        gmm = GaussianMixture(n_components=4, covariance_type='spherical',
                              random_state=42, reg_covar=1e-6)
        gmm.fit(feats_c)

        dists = np.linalg.norm(feats_c[:, None] - feats_c[None, :], axis=-1)
        avg_dist = dists[dists > 0].mean()

        print(f"  class {int(cid):3d} ({c_names[cid]:30s} n={len(feats_c)}): "
              f"sigma2={[f'{v:.4f}' for v in gmm.covariances_]} "
              f"avg_pair_dist={avg_dist:.3f}")
        vars_for_dataset.extend(gmm.covariances_)
    all_vars.extend(vars_for_dataset)
    print(f"  -> mean sigma2 for this dataset: {np.mean(vars_for_dataset):.6f}, "
          f"sigma={np.sqrt(np.mean(vars_for_dataset)):.4f}")

print(f"\n=== OVERALL ===")
print(f"Total components: {len(all_vars)}")
print(f"mean sigma2: {np.mean(all_vars):.6f}")
print(f"sigma (sqrt mean): {np.sqrt(np.mean(all_vars)):.4f}")
print(f"min sigma2: {np.min(all_vars):.6f}, max sigma2: {np.max(all_vars):.6f}")
print(f"mean of sqrt(sigma2): {np.mean(np.sqrt(all_vars)):.4f}")
