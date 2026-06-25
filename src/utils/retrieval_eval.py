import base64
import csv
import io
import json
import ast
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils_data import Flickr8kDataset, get_transforms


class RetrievalImageDataset(Dataset):
    def __init__(self, retrieval_dataset):
        self.retrieval_dataset = retrieval_dataset

    def __len__(self):
        return len(self.retrieval_dataset.samples)

    def __getitem__(self, idx):
        sample = self.retrieval_dataset.samples[idx]
        image = self.retrieval_dataset.open_image(sample)
        image = self.retrieval_dataset.transform(image)
        return image, idx


class PathCaptionRetrievalDataset:
    def __init__(self, name, root, samples, prompts_list, transform, language="en"):
        self.name = name
        self.root = str(root)
        self.samples = samples
        self.prompts_list = prompts_list
        self.transform = transform
        self.language = language

    def __len__(self):
        return len(self.samples)

    def open_image(self, sample):
        return Image.open(sample["path"]).convert("RGB")

    def metadata(self):
        return {
            "name": self.name,
            "root": self.root,
            "language": self.language,
            "num_images": len(self.samples),
            "num_captions": sum(len(captions) for captions in self.prompts_list),
        }


class Base64CaptionRetrievalDataset(PathCaptionRetrievalDataset):
    def open_image(self, sample):
        raw = base64.b64decode(sample["image_base64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")


class BytesCaptionRetrievalDataset(PathCaptionRetrievalDataset):
    def open_image(self, sample):
        return Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")


def parse_recall_ks(raw):
    if isinstance(raw, (list, tuple)):
        ks = sorted({int(item) for item in raw})
    else:
        ks = sorted({int(item) for item in str(raw).replace(" ", "").split(",") if item})
    if not ks:
        raise ValueError("retrieval recall ks produced no values")
    return ks


def parse_retrieval_roots(raw):
    roots = {}
    if not raw:
        return roots
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid retrieval root entry {item!r}; expected dataset=/path"
            )
        name, value = item.split("=", 1)
        roots[name.strip()] = Path(value.strip())
    return roots


def _standard_transform():
    _, transform_test = get_transforms("flickr8k")
    return transform_test


def _apply_max_images(samples, prompts_list, max_images):
    if max_images and int(max_images) > 0:
        keep = min(int(max_images), len(samples))
        return samples[:keep], prompts_list[:keep]
    return samples, prompts_list


def load_retrieval_dataset(dataset_name, root, max_images=0):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Retrieval dataset root does not exist: {root}")

    transform = _standard_transform()
    if dataset_name == "flickr8k":
        dataset = Flickr8kDataset(str(root), transform=transform)
        if max_images and int(max_images) > 0:
            dataset.samples = dataset.samples[: int(max_images)]
            dataset.prompts_list = dataset.prompts_list[: int(max_images)]
            print(f"[RetrievalEval] flickr8k using first {len(dataset.samples)} images", flush=True)
        dataset.name = "flickr8k"
        dataset.root = str(root)
        dataset.language = "en"
        dataset.open_image = lambda sample: Image.open(sample[0]).convert("RGB")
        dataset.metadata = lambda: {
            "name": "flickr8k",
            "root": str(root),
            "language": "en",
            "num_images": len(dataset.samples),
            "num_captions": sum(len(captions) for captions in dataset.prompts_list),
        }
        return dataset

    if dataset_name == "coco_val2014":
        images_dir = root / "val2014"
        ann_path = root / "annotations" / "captions_val2014.json"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"COCO val2014 image directory missing: {images_dir}")
        if not ann_path.is_file():
            raise FileNotFoundError(f"COCO captions annotation missing: {ann_path}")

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        id_to_file = {int(item["id"]): item["file_name"] for item in data.get("images", [])}
        captions_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            image_id = int(ann["image_id"])
            caption = str(ann.get("caption", "")).strip()
            if caption:
                captions_by_image[image_id].append(caption)

        samples = []
        prompts_list = []
        for image_id in sorted(captions_by_image):
            file_name = id_to_file.get(image_id)
            if not file_name:
                continue
            path = images_dir / file_name
            if not path.is_file():
                continue
            samples.append({"image_id": image_id, "path": path})
            prompts_list.append(captions_by_image[image_id])
        samples, prompts_list = _apply_max_images(samples, prompts_list, max_images)
        if not samples:
            raise ValueError(f"No COCO retrieval samples found under {root}")
        return PathCaptionRetrievalDataset(
            "coco_val2014", root, samples, prompts_list, transform, language="en"
        )

    if dataset_name == "mscoco_2014_5k":
        images_dir = root / "images_mscoco_2014_5k_test"
        csv_path = root / "test_5k_mscoco_2014.csv"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"MSCOCO 5K image directory missing: {images_dir}")
        if not csv_path.is_file():
            raise FileNotFoundError(f"MSCOCO 5K CSV missing: {csv_path}")

        samples = []
        prompts_list = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = row.get("filename", "").strip()
                if not filename:
                    continue
                path = images_dir / filename
                if not path.is_file():
                    continue
                raw_captions = ast.literal_eval(row.get("raw", "[]"))
                captions = [str(caption).strip() for caption in raw_captions if str(caption).strip()]
                if not captions:
                    continue
                samples.append({"image_id": row.get("cocoid", filename), "path": path})
                prompts_list.append(captions)
        samples, prompts_list = _apply_max_images(samples, prompts_list, max_images)
        if not samples:
            raise ValueError(f"No MSCOCO 5K retrieval samples found under {root}")
        return PathCaptionRetrievalDataset(
            "mscoco_2014_5k", root, samples, prompts_list, transform, language="en"
        )

    if dataset_name == "flickr30k_hf":
        data_dir = root / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Flickr30K HF data directory missing: {data_dir}")
        parquet_paths = sorted(data_dir.glob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No Flickr30K HF parquet files found under {data_dir}")

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("flickr30k_hf loader requires pandas with parquet support") from exc

        samples = []
        prompts_list = []
        limit = int(max_images) if max_images and int(max_images) > 0 else None
        for parquet_path in parquet_paths:
            frame = pd.read_parquet(
                parquet_path,
                columns=["image", "filename", "original_alt_text", "alt_text"],
            )
            for _, row in frame.iterrows():
                image_obj = row["image"]
                if not isinstance(image_obj, dict) or not image_obj.get("bytes"):
                    continue
                captions_obj = row["original_alt_text"]
                if captions_obj is None or len(captions_obj) == 0:
                    captions_obj = row["alt_text"]
                captions = [str(caption).strip() for caption in captions_obj if str(caption).strip()]
                if not captions:
                    continue
                filename = str(row.get("filename") or image_obj.get("path") or len(samples))
                samples.append({
                    "image_id": filename,
                    "filename": filename,
                    "image_bytes": image_obj["bytes"],
                })
                prompts_list.append(captions)
                if limit is not None and len(samples) >= limit:
                    break
            if limit is not None and len(samples) >= limit:
                break
        if not samples:
            raise ValueError(f"No Flickr30K HF retrieval samples found under {root}")
        return BytesCaptionRetrievalDataset(
            "flickr30k_hf", root, samples, prompts_list, transform, language="en"
        )

    if dataset_name == "flickr30k_cn":
        imgs_path = root / "test_imgs.tsv"
        texts_path = root / "test_texts.jsonl"
        if not imgs_path.is_file():
            raise FileNotFoundError(f"Flickr30k-CN image TSV missing: {imgs_path}")
        if not texts_path.is_file():
            raise FileNotFoundError(f"Flickr30k-CN text jsonl missing: {texts_path}")

        captions_by_image = defaultdict(list)
        with texts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                caption = str(item.get("text", "")).strip()
                if not caption:
                    continue
                for image_id in item.get("image_ids", []):
                    captions_by_image[str(image_id)].append(caption)

        samples = []
        prompts_list = []
        limit = int(max_images) if max_images and int(max_images) > 0 else None
        with imgs_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                image_id = row[0].strip()
                if image_id not in captions_by_image:
                    continue
                samples.append({"image_id": image_id, "image_base64": row[1].strip()})
                prompts_list.append(captions_by_image[image_id])
                if limit is not None and len(samples) >= limit:
                    break
        if not samples:
            raise ValueError(f"No Flickr30k-CN retrieval samples found under {root}")
        return Base64CaptionRetrievalDataset(
            "flickr30k_cn", root, samples, prompts_list, transform, language="zh"
        )

    raise ValueError(
        f"Unsupported retrieval dataset {dataset_name!r}. "
        "Supported: coco_val2014, mscoco_2014_5k, flickr30k_hf, flickr30k_cn, flickr8k"
    )


@torch.no_grad()
def encode_images(model, dataset, device, batch_size, num_workers):
    loader = DataLoader(
        RetrievalImageDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    features = []
    seen_indices = []
    for images, indices in tqdm(loader, desc="Encoding retrieval images"):
        images = images.to(device)
        outputs = model.vision_model(images)
        pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs[1]
        feats = model.visual_projection(pooled)
        feats = F.normalize(feats.float(), dim=-1)
        features.append(feats.cpu())
        seen_indices.append(indices.cpu())

    features = torch.cat(features, dim=0)
    seen_indices = torch.cat(seen_indices, dim=0)
    if not torch.equal(seen_indices, torch.arange(len(dataset))):
        raise RuntimeError("Image dataloader changed retrieval order")
    return features


@torch.no_grad()
def encode_texts(model, processor, captions, device, batch_size):
    features = []
    for start in tqdm(range(0, len(captions), batch_size), desc="Encoding retrieval captions"):
        end = min(start + batch_size, len(captions))
        inputs = processor(
            text=captions[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.text_model(**inputs)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state[:, -1, :]
        else:
            pooled = outputs[1] if isinstance(outputs, tuple) else outputs
        feats = model.text_projection(pooled)
        feats = F.normalize(feats.float(), dim=-1)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)


def build_caption_index(dataset):
    captions = []
    image_to_text = []
    text_to_image = []
    for image_idx, image_captions in enumerate(dataset.prompts_list):
        indices = []
        for caption in image_captions:
            indices.append(len(captions))
            captions.append(caption)
            text_to_image.append(image_idx)
        image_to_text.append(set(indices))
    return captions, image_to_text, torch.tensor(text_to_image, dtype=torch.long)


def _recall_dict(correct_counts, total, ks):
    return {f"r@{k}": (correct_counts[k] / total * 100.0 if total else 0.0) for k in ks}


@torch.no_grad()
def compute_retrieval_metrics(image_features, text_features, image_to_text, text_to_image, ks, chunk_size):
    image_features = image_features.float()
    text_features = text_features.float()
    max_k = max(ks)
    i2t_correct = {k: 0 for k in ks}
    t2i_correct = {k: 0 for k in ks}

    for start in tqdm(range(0, image_features.shape[0], chunk_size), desc="I2T retrieval"):
        end = min(start + chunk_size, image_features.shape[0])
        sims = image_features[start:end] @ text_features.t()
        topk = sims.topk(k=min(max_k, text_features.shape[0]), dim=1).indices
        for row_idx, caption_indices in enumerate(topk):
            positives = image_to_text[start + row_idx]
            preds = caption_indices.tolist()
            for k in ks:
                if any(pred in positives for pred in preds[:k]):
                    i2t_correct[k] += 1

    for start in tqdm(range(0, text_features.shape[0], chunk_size), desc="T2I retrieval"):
        end = min(start + chunk_size, text_features.shape[0])
        sims = text_features[start:end] @ image_features.t()
        topk = sims.topk(k=min(max_k, image_features.shape[0]), dim=1).indices
        targets = text_to_image[start:end]
        for row_idx, image_indices in enumerate(topk):
            preds = image_indices.tolist()
            target = int(targets[row_idx].item())
            for k in ks:
                if target in preds[:k]:
                    t2i_correct[k] += 1

    return {
        "i2t": _recall_dict(i2t_correct, image_features.shape[0], ks),
        "t2i": _recall_dict(t2i_correct, text_features.shape[0], ks),
    }


def evaluate_retrieval_model(
    model,
    processor,
    dataset,
    device,
    batch_size=128,
    text_batch_size=256,
    num_workers=4,
    recall_ks=(1, 5, 10),
    similarity_chunk_size=512,
):
    ks = parse_recall_ks(recall_ks)
    image_features = encode_images(model, dataset, device, batch_size, num_workers)
    captions, image_to_text, text_to_image = build_caption_index(dataset)
    text_features = encode_texts(model, processor, captions, device, text_batch_size)
    metrics = compute_retrieval_metrics(
        image_features,
        text_features,
        image_to_text,
        text_to_image,
        ks,
        int(similarity_chunk_size),
    )
    metrics["num_images"] = int(image_features.shape[0])
    metrics["num_captions"] = int(text_features.shape[0])
    return metrics


def retrieval_payload(method, dataset, metrics, args=None, artifact_path=None, step_index=None, task=None):
    payload = {
        "schema_version": 1,
        "method": method,
        "dataset": dataset.name,
        "dataset_meta": dataset.metadata(),
        "metrics": metrics,
    }
    if artifact_path is not None:
        payload["artifact_path"] = str(artifact_path)
    if step_index is not None:
        payload["step_index"] = int(step_index)
    if task is not None:
        payload["task"] = task
    if args is not None:
        payload["args"] = args
    return payload


def flatten_retrieval_row(method, step, task, dataset_name, metrics, path):
    return {
        "method": method,
        "step": int(step),
        "task": task,
        "dataset": dataset_name,
        "num_images": metrics["num_images"],
        "num_captions": metrics["num_captions"],
        "i2t_r@1": metrics["i2t"].get("r@1", 0.0),
        "i2t_r@5": metrics["i2t"].get("r@5", 0.0),
        "i2t_r@10": metrics["i2t"].get("r@10", 0.0),
        "t2i_r@1": metrics["t2i"].get("r@1", 0.0),
        "t2i_r@5": metrics["t2i"].get("r@5", 0.0),
        "t2i_r@10": metrics["t2i"].get("r@10", 0.0),
        "path": str(path),
    }
