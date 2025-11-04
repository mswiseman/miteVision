import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision.ops import box_iou
import torchvision
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO = True
except Exception:
    HAS_COCO = False


# -----------------------------
# Utils
# -----------------------------

def seed_everything(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def read_data_yaml(data_yaml_path: str) -> Dict:
    import yaml
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Expected keys: train, val, nc, names
    return data


def yolo_txt_to_xyxy(label_path: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    YOLO format: class cx cy w h (normalized 0-1).
    Returns boxes in absolute xyxy and labels in int64.
    """
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes = []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            c, cx, cy, w, h = map(float, parts)
            c = int(c)
            # denormalize
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_w - 1, x2); y2 = min(img_h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(c + 1)  # torchvision needs labels in [1..nc]
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


class YOLODetectionDataset(Dataset):
    """
    Minimal dataset to feed Faster R-CNN using YOLO-format labels.
    """
    def __init__(self, images_dir: str):
        self.images = []
        p = Path(images_dir)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        for im in sorted(p.rglob("*")):
            if im.suffix.lower() in exts:
                self.images.append(im)
        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found under {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = img_path.with_suffix(".txt").parent.parent / "labels" / img_path.name.replace(img_path.suffix, ".txt")
        # Support common structure .../images/{train|val}/xxx.jpg and .../labels/{train|val}/xxx.txt
        if not lbl_path.exists():
            # Fallback: label next to image with .txt
            lbl_path = img_path.with_suffix(".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes, labels = yolo_txt_to_xyxy(lbl_path, w, h)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "area": torch.as_tensor(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) if len(boxes) else np.zeros((0,)), dtype=torch.float32),
            "orig_size": torch.as_tensor([h, w], dtype=torch.int64),
            "size": torch.as_tensor([h, w], dtype=torch.int64),
        }

        # Basic transforms: convert to tensor, do not resize (Faster R-CNN handles internally)
        img_tensor = to_tensor(img)
        return img_tensor, target


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


def coco_eval_from_results(cocoGt, results):
    cocoDt = cocoGt.loadRes(results)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    stats = cocoEval.stats  # [AP, AP50, AP75, AP_s, AP_m, AP_l, AR...]
    return {
        "mAP50-95": float(stats[0]),
        "mAP50": float(stats[1]),
    }


def to_coco_json(det_outputs, dataset: YOLODetectionDataset, class_map: Dict[int, str]):
    """
    Convert predictions to COCO detection JSON list.
    class_map expects {1:"cls0", 2:"cls1", ...}
    """
    results = []
    for (img_id, det) in det_outputs:
        boxes = det["boxes"].cpu().numpy()
        scores = det["scores"].cpu().numpy()
        labels = det["labels"].cpu().numpy()
        for b, s, c in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b.tolist()
            w = x2 - x1
            h = y2 - y1
            results.append({
                "image_id": int(img_id),
                "category_id": int(c),  # categories start at 1
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(s),
            })
    categories = [{"id": i, "name": class_map[i]} for i in sorted(class_map)]
    # Build COCO-style GT to evaluate properly
    # For convenience we will write a minimal COCO GT file from the YOLO dataset on-the-fly.
    return results


def build_min_coco_gt(dataset: YOLODetectionDataset, class_map: Dict[int, str], out_json: Path):
    """
    Create a minimal COCO GT JSON from YOLO labels for evaluation.
    Includes 'info' and 'licenses' so pycocotools.loadRes() is happy.
    """
    import json
    from PIL import Image

    images = []
    annotations = []
    ann_id = 1

    for idx, img_path in enumerate(dataset.images):
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({
            "id": idx,
            "file_name": str(img_path.name),  # names are fine; COCO uses id for matching anyway
            "width": w,
            "height": h
        })

        lbl_path = img_path.with_suffix(".txt").parent.parent / "labels" / img_path.name.replace(img_path.suffix, ".txt")
        if not lbl_path.exists():
            lbl_path = img_path.with_suffix(".txt")

        boxes, labels = yolo_txt_to_xyxy(lbl_path, w, h)
        for b, c in zip(boxes, labels):
            x1, y1, x2, y2 = b.tolist()
            bw = x2 - x1
            bh = y2 - y1
            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": int(c),  # categories start at 1
                "bbox": [float(x1), float(y1), float(bw), float(bh)],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i, "name": class_map[i]} for i in sorted(class_map)]

    coco_dict = {
        "info": {
            "description": "YOLO->COCO minimal GT",
            "version": "1.0",
            "year": 2025
        },
        "licenses": [{"id": 1, "name": "unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco_dict, f)
    return out_json


def params_count(model) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return None


def speed_bench_callable(predict_callable, imgsz=640, device="cuda", runs=30) -> Tuple[float, float, float]:
    """
    Benchmark a callable that takes a pre-allocated tensor batch and performs one forward.
    Returns mean_ms, std_ms, fps.
    """
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    # warmup
    for _ in range(8):
        _ = predict_callable(dummy)
    times = []
    for _ in range(runs):
        if "cuda" in device:
            torch.cuda.synchronize()
        t0 = time.time()
        _ = predict_callable(dummy)
        if "cuda" in device:
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000.0)
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return mean_ms, std_ms, fps


# -----------------------------
# YOLO branch (Ultralytics)
# -----------------------------

def train_eval_yolo(model_name, data_yaml, epochs, imgsz, batch, device, workers, project, prefix) -> Dict:
    print(f"\n==== Training {model_name} (Ultralytics) ====")
    model = YOLO(model_name)
    train_res = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=f"{prefix}_{Path(model_name).stem}",
        verbose=True,
    )
    print(f"\n==== Validating {model_name} (Ultralytics) ====")
    val_res = model.val(imgsz=imgsz, device=device, verbose=False)
    metrics = val_res.results_dict

    box_map = float(metrics.get("metrics/mAP50-95(B)", float("nan")))
    box_map50 = float(metrics.get("metrics/mAP50(B)", float("nan")))
    box_precision = float(metrics.get("metrics/precision(B)", float("nan")))
    box_recall = float(metrics.get("metrics/recall(B)", float("nan")))

    pcount = params_count(model.model)

    def _ultra_pred(x):
        with torch.inference_mode():
            _ = model.predict(source=x, imgsz=imgsz, verbose=False)

    mean_ms, std_ms, fps = speed_bench_callable(_ultra_pred, imgsz=imgsz, device=device)

    row = {
        "framework": "Ultralytics",
        "model": model_name,
        "params": pcount,
        "val/mAP50-95": box_map,
        "val/mAP50": box_map50,
        "val/precision": box_precision,
        "val/recall": box_recall,
        "infer_ms_per_img": mean_ms,
        "infer_ms_std": std_ms,
        "approx_FPS": fps,
        "run_dir": str(model.trainer.save_dir) if hasattr(model, "trainer") else "",
    }
    
    return row


# -----------------------------
# Faster R-CNN branch (torchvision)
# -----------------------------

def get_frcnn_model(num_classes: int):
    """
    Build Faster R-CNN (ResNet50-FPNv2). num_classes includes background.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    # Replace head for num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    new_head = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.box_predictor = new_head
    return model


@torch.no_grad()
def evaluate_simple_map_at_05(model, data_loader, device="cuda", score_thr=0.001, iou_thr=0.5):
    """
    Simple mAP@0.5 (11-point) approximation for quick sanity. Use COCO eval if pycocotools available.
    """
    model.eval()
    aps = []
    for imgs, targets in data_loader:
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)
        for out, tgt in zip(outputs, targets):
            gt_boxes = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device)
            sel = out["scores"] >= score_thr
            boxes = out["boxes"][sel]
            scores = out["scores"][sel]
            labels = out["labels"][sel]

            # class-wise matching
            classes = torch.unique(torch.cat([labels, gt_labels])).tolist()
            for c in classes:
                pred_idx = labels == c
                gt_idx = gt_labels == c
                if gt_idx.sum() == 0:
                    continue
                biou = box_iou(boxes[pred_idx], gt_boxes[gt_idx])
                # greedy match
                matched_gts = set()
                tp = []
                conf = []
                for j in torch.argsort(scores[pred_idx], descending=True):
                    j = int(j)
                    if biou.shape[0] == 0:
                        tp.append(0.0); conf.append(float(scores[pred_idx][j])); continue
                    i = int(torch.argmax(biou[j]))
                    iou = float(biou[j, i])
                    if iou >= iou_thr and i not in matched_gts:
                        tp.append(1.0)
                        matched_gts.add(i)
                    else:
                        tp.append(0.0)
                    conf.append(float(scores[pred_idx][j]))
                if len(tp):
                    # 11-point AP
                    tp = np.array(tp)
                    fp = 1 - tp
                    cum_tp = np.cumsum(tp)
                    cum_fp = np.cumsum(fp)
                    recalls = cum_tp / max(1, gt_idx.sum().item())
                    precisions = cum_tp / np.maximum(1, (cum_tp + cum_fp))
                    # interpolate at r = 0.0..1.0 step=0.1
                    ap = 0.0
                    for r in np.linspace(0, 1, 11):
                        p = np.max(precisions[recalls >= r]) if np.any(recalls >= r) else 0.0
                        ap += p / 11.0
                    aps.append(ap)
    mAP = float(np.mean(aps)) if len(aps) else 0.0
    return {"mAP@0.5_simple": mAP}


def train_eval_frcnn(
    train_dir, val_dir, nc, class_names, epochs, batch_size, device, workers, imgsz, project, prefix
) -> Dict:
    """
    Train Faster R-CNN on YOLO dataset directories + rich prints per epoch.
    """
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    print("[FRCNN] Building datasets...")
    ds_train = YOLODetectionDataset(train_dir)
    ds_val = YOLODetectionDataset(val_dir)
    print(f"[FRCNN] Train images: {len(ds_train)} | Val images: {len(ds_val)}")
    print(f"[FRCNN] Device: {device} | Epochs: {epochs} | Batch size: {batch_size}")

    num_classes = nc + 1  # include background
    model = get_frcnn_model(num_classes)
    model.to(device)

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False,
                        num_workers=workers, collate_fn=collate_fn, pin_memory=True)

    # Optimizer & LR schedule
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.7*epochs), int(0.9*epochs)], gamma=0.1)

    save_dir = Path(project) / f"{prefix}_fasterrcnn_r50fpn"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[FRCNN] Starting training...")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        losses = []

        # ---- Training loop ----
        for imgs, targets in dl_train:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        lr_sched.step()
        epoch_time = time.time() - epoch_start
        mean_loss = float(np.mean(losses)) if len(losses) else float("nan")
        print(f"[FRCNN] Epoch {epoch+1:03d}/{epochs} | Loss={mean_loss:.4f} | Time={epoch_time/60:.2f} min")

        # ---- Quick validation every few epochs ----
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:  # every 5 epochs or last
            model.eval()
            with torch.inference_mode():
                aps = []
                for imgs, targets in dl_val:
                    imgs = [img.to(device) for img in imgs]
                    outputs = model(imgs)
                    for out, tgt in zip(outputs, targets):
                        gt_boxes = tgt["boxes"].to(device)
                        if gt_boxes.numel() == 0:
                            continue
                        sel = out["scores"] >= 0.05
                        boxes = out["boxes"][sel]
                        if boxes.numel() == 0:
                            continue
                        biou = box_iou(boxes, gt_boxes)
                        max_iou, _ = torch.max(biou, dim=1)
                        # crude per-image AP@0.5 proxy: fraction of preds with IoU>=0.5
                        aps.append(float((max_iou >= 0.5).float().mean()))
                val_map50 = float(np.mean(aps)) if aps else 0.0
            print(f"[FRCNN] └── Validation: mAP@0.5 ≈ {val_map50:.4f}")

    # ---- Post-training speed benchmark ----
    print("[FRCNN] Evaluating on val set...")
    def _frcnn_forward(x):
        with torch.inference_mode():
            _ = model([x[0]])
    mean_ms, std_ms, fps = speed_bench_callable(_frcnn_forward, imgsz=imgsz, device=device)

    # ---- Final metrics (COCO if available) ----
    class_map = {i+1: n for i, n in enumerate(class_names)}
    if HAS_COCO:
        print("[FRCNN] Running COCO evaluation (pycocotools)...")
        coco_gt_json = save_dir / "val_gt_coco.json"
        build_min_coco_gt(ds_val, class_map, coco_gt_json)
        cocoGt = COCO(str(coco_gt_json))

        det_outputs = []
        with torch.inference_mode():
            for imgs, targets in dl_val:
                imgs = [img.to(device) for img in imgs]
                outputs = model(imgs)
                for tgt, out in zip(targets, outputs):
                    det_outputs.append((tgt["image_id"].item(), out))

        results_json = []
        for (img_id, det) in det_outputs:
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            for b, s, c in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b.tolist()
                w = x2 - x1
                h = y2 - y1
                results_json.append({
                    "image_id": int(img_id),
                    "category_id": int(c),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(s),
                })
        pred_json_path = save_dir / "val_preds_coco.json"
        with open(pred_json_path, "w") as f:
            json.dump(results_json, f)

        cocoEval = COCOeval(cocoGt, cocoGt.loadRes(str(pred_json_path)), iouType='bbox')
        cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
        stats = cocoEval.stats
        mAP_50_95 = float(stats[0]); mAP_50 = float(stats[1])
        print(f"[FRCNN] COCO mAP50-95={mAP_50_95:.4f} | mAP50={mAP_50:.4f}")
    else:
        print("[FRCNN] pycocotools not installed; using simple mAP@0.5 estimator.")
        m = evaluate_simple_map_at_05(model, dl_val, device=device)
        mAP_50_95 = float("nan"); mAP_50 = float(m["mAP@0.5_simple"])
        print(f"[FRCNN] simple mAP@0.5={mAP_50:.4f}")

    # ---- Save weights ----
    weights_path = save_dir / "fasterrcnn_resnet50_fpn_v2_final.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"[FRCNN] Saved model weights to: {weights_path}")

    row = {
        "framework": "Torchvision",
        "model": "fasterrcnn_resnet50_fpn_v2",
        "params": params_count(model),
        "val/mAP50-95": mAP_50_95,
        "val/mAP50": mAP_50,
        "val/precision": float("nan"),
        "val/recall": float("nan"),
        "infer_ms_per_img": mean_ms,
        "infer_ms_std": std_ms,
        "approx_FPS": fps,
        "run_dir": str(save_dir),
    }
    print(f"[FRCNN] Artifacts saved to: {save_dir}")
    return row


# -----------------------------
# Orchestration
# -----------------------------

def main():
    parser = argparse.ArgumentParser("Compare YOLO (Ultralytics) and Faster R-CNN on a YOLO dataset")
    parser.add_argument("--data", required=True, help="Path to YOLO data YAML")
    parser.add_argument("--models", nargs="+", default=[], help="Ultralytics YOLO models to train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto for Ultralytics; used as batch_size for FRCNN")
    parser.add_argument("--device", default="cuda", help="'cuda', '0', 'cpu', etc.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/compare")
    parser.add_argument("--name_prefix", default="exp")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    data = read_data_yaml(args.data)
    nc = int(data["nc"])
    names = data["names"]
    # Resolve train/val images dirs from YAML; they can be txt files listing paths or a directory
    train = data["train"]
    val = data["val"]

    # If YAML points to files listing paths, assume images are in ".../images/{train|val}"
    def infer_images_dir(x):
        x = str(x)
        if os.path.isdir(x):
            return x
        if os.path.isfile(x):
            # if it's a text file of image paths, we try to use the parent 'images' folder
            p = Path(x).parent
            return str(p)
        return x

    train_images_dir = infer_images_dir(train)
    val_images_dir = infer_images_dir(val)

    project_path = Path(args.project)
    project_path.mkdir(parents=True, exist_ok=True)

    rows = []

    # YOLO models
    for m in args.models:
        row = train_eval_yolo(
            model_name=m,
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            workers=args.workers,
            project=args.project,
            prefix=args.name_prefix
        )
        print(json.dumps(row, indent=2))
        rows.append(row)

    # Faster R-CNN (Torchvision)
    fr_row = train_eval_frcnn(
        train_dir=train_images_dir,
        val_dir=val_images_dir,
        nc=nc,
        class_names=names,
        epochs=args.epochs,
        batch_size=(args.batch if args.batch != -1 else 2),
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        workers=args.workers,
        imgsz=args.imgsz,
        project=args.project,
        prefix=args.name_prefix
    )
    print(json.dumps(fr_row, indent=2))
    rows.append(fr_row)

    # Save comparison CSV
    df = pd.DataFrame(rows)
    out_csv = project_path / "comparison_yolo_frcnn.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved comparison table to: {out_csv.resolve()}\n")
    print(df.sort_values(["val/mAP50-95"], ascending=False))


if __name__ == "__main__":
    import sys
    sys.argv = [
        "notebook",
        "--data", "/content/drive/MyDrive/blackbird/mite_detection/TSSM-Detection-v2-209/data.yaml",
        "--models", "yolo11n.pt", "yolo11l.pt", "yolov8n.pt", "yolov8l.pt",
        "--epochs", "50",
        "--imgsz", "1024",
        "--batch", "-1",
        "--device", "cuda",
        "--workers", "2"
    ]
    main()
