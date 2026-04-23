# %% [markdown]
# # Leaf Segmenter — Training Script (Fixed + Improved)
#
# Trains DeepLabV3 + MobileNetV3-Large to separate rice leaf tissue from
# everything else (hand, soil, sky, weeds, tools).
#
# Key fixes vs the original:
#   - BUG FIX: The original script called ds_trn.dataset.transforms = ...
#     which modified the SHARED underlying dataset object, applying train
#     augmentation to validation images too. Fixed by creating two independent
#     Dataset instances (one per split) with their own transform.
#   - Warm-start from existing checkpoint instead of always starting from
#     COCO weights — saves time when iteratively adding labels.
#   - Dice + CrossEntropy combined loss — Dice is better for thin/narrow
#     structures like rice leaf blades.
#   - Confidence map export after training — shows WHERE the model is
#     uncertain so you can label those images next.
#   - Proper per-class IoU logging (leaf IoU and background IoU separately).
#
# ## Folder layout
# ```
# TL_Data/
#   leaves/
#     images/   <- JPEG or PNG, any resolution
#     masks/    <- uint8 PNG, 0=background, 1 (or 255)=leaf, same stem .png
# TL_Data_eval/
#   Leaves/
#     Images/   <- unlabelled images to run inference on
# models/
#   leaf_segmenter_best.pt   <- warm-started if present
# ```
#
# ## Quick start
# ```bash
# pip install torch torchvision albumentations opencv-python tqdm pandas
# python train_leaf_segmenter.py
# ```

# %% Cell 1 — Imports and config
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

CFG = {
    "data_root":        Path("TL_Data/leaves"),   # images/ and masks/ here
    "out_dir":          Path("models"),
    "img_size":         512,
    "batch_size":       4,          # safe for 4 GB VRAM; try 8 if you have 8 GB
    "num_workers":      2,
    "epochs":           50,         # warm-start means you can always add more
    "lr":               5e-5,
    "weight_decay":     1e-4,
    "val_frac":         0.2,
    "seed":             42,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes":      2,          # 0 = background, 1 = leaf
    "warmstart":        True,       # load existing checkpoint if present
    "export_conf_maps": True,       # write uncertainty PNGs after training
}
CFG["out_dir"].mkdir(parents=True, exist_ok=True)
torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])
print(f"Device: {CFG['device']}")
print(f"Data root: {CFG['data_root'].resolve()}")


# %% Cell 2 — Dataset
class LeafSegDataset(Dataset):
    """
    Pairs each image with its binary leaf mask.
    Masks: uint8 PNG — 0 = background, any non-zero = leaf.
    Each Dataset instance has its own transform so train/val never share state.
    """

    def __init__(self, root: Path, transforms=None):
        self.img_paths = sorted((root / "images").glob("*.*"))
        self.mask_dir = root / "masks"
        self.transforms = transforms
        missing = [p.name for p in self.img_paths
                   if not (self.mask_dir / (p.stem + ".png")).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} mask(s) missing. First few: {missing[:5]}")
        print(f"  Dataset: {len(self.img_paths)} image-mask pairs found.")

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.int64)
        if self.transforms:
            out = self.transforms(image=img, mask=mask.astype(np.uint8))
            img = out["image"]
            mask = out["mask"].long()
        return img, mask, img_path.name


# %% Cell 3 — Proper train/val split (KEY FIX — two separate Dataset objects)
def make_split_datasets(root, img_size, val_frac, seed):
    """
    Creates two independent Dataset instances with their own transforms.
    This is the correct way to split — sharing the same dataset object
    and setting .transforms afterwards mutates BOTH splits.
    """
    # Dummy read to get file count and compute split indices.
    dummy = LeafSegDataset(root, transforms=None)
    n = len(dummy)
    n_val = max(1, int(n * val_frac))
    n_trn = n - n_val
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n).tolist()
    trn_idx = idx[:n_trn]
    val_idx = idx[n_trn:]

    # Two fresh instances — each owns its own transform.
    ds_trn = LeafSegDataset(
        root, transforms=build_transforms(img_size, train=True))
    ds_val = LeafSegDataset(
        root, transforms=build_transforms(img_size, train=False))
    return Subset(ds_trn, trn_idx), Subset(ds_val, val_idx)


# %% Cell 4 — Augmentations
def build_transforms(img_size, train=True):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if train:
        return A.Compose([
            A.LongestMaxSize(img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                               rotate_limit=30, p=0.6,
                               border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
            A.HueSaturationValue(12, 25, 12, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.25),
            A.GaussNoise(p=0.15),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32,
                            fill_value=0, p=0.2),
            A.Normalize(mean, std),
            ToTensorV2(),
        ])
    return A.Compose([
        A.LongestMaxSize(img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])


# %% Cell 5 — Model factory (importable by other scripts)
def build_model(num_classes=2):
    """DeepLabV3 + MobileNetV3-Large, COCO-pretrained, head replaced."""
    weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(10, num_classes, kernel_size=1)
    return model


# %% Cell 6 — Combined Dice + CrossEntropy loss
class DiceCELoss(nn.Module):
    """
    0.5 * CrossEntropy + 0.5 * Dice.
    Dice handles thin/small structure recall; CE handles calibration.
    """

    def __init__(self, ce_weights=None, smooth=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weights)
        self.smooth = smooth

    def forward(self, logits, target):
        ce_l = self.ce(logits, target)
        probs = F.softmax(logits, dim=1)[:, 1]
        t_bin = (target == 1).float()
        inter = (probs * t_bin).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + t_bin.sum(dim=(1, 2))
        dice = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return 0.5 * ce_l + 0.5 * dice.mean()


# %% Cell 7 — Metrics
def iou_score(logits, target, num_classes=2):
    """Returns (mean_iou, {class_name: iou}) dict."""
    pred = logits.argmax(dim=1)
    names = {0: "background", 1: "leaf"}
    ious, per_class = [], {}
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().item()
        union = ((pred == c) | (target == c)).sum().item()
        if union > 0:
            iou = inter / union
            ious.append(iou)
            per_class[names.get(c, str(c))] = iou
    return (float(np.mean(ious)) if ious else 0.0), per_class


# %% Cell 8 — Training loop
def train():
    ds_trn, ds_val = make_split_datasets(
        CFG["data_root"], CFG["img_size"], CFG["val_frac"], CFG["seed"])
    print(f"  Train: {len(ds_trn)}   Val: {len(ds_val)}")

    dl_trn = DataLoader(ds_trn, batch_size=CFG["batch_size"], shuffle=True,
                        num_workers=CFG["num_workers"], pin_memory=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True)

    model = build_model(CFG["num_classes"]).to(CFG["device"])
    ckpt_path = CFG["out_dir"] / "leaf_segmenter_best.pt"
    best_iou = 0.0

    # --- Warm-start ---
    if CFG["warmstart"] and ckpt_path.exists():
        ckpt = torch.load(
            ckpt_path, map_location=CFG["device"], weights_only=False)
        try:
            model.load_state_dict(ckpt["model"])
            best_iou = float(ckpt.get("iou", 0.0))
            print(
                f"  Warm-start from checkpoint (prev best mIoU={best_iou:.3f})")
        except RuntimeError as e:
            print(f"  [WARN] Could not warm-start: {e}\n  Using COCO weights.")
    else:
        print("  No checkpoint — starting from COCO pretrained weights.")

    opt = torch.optim.AdamW(model.parameters(),
                            lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=CFG["epochs"])
    ce_w = torch.tensor([1.0, 3.0], device=CFG["device"])
    loss_fn = DiceCELoss(ce_weights=ce_w)
    history = []

    for epoch in range(CFG["epochs"]):
        # Train
        model.train()
        trn_loss = 0.0
        pbar = tqdm(dl_trn, desc=f"Epoch {epoch+1:02d}/{CFG['epochs']} [train]",
                    leave=False)
        for imgs, masks, _ in pbar:
            imgs = imgs.to(CFG["device"], non_blocking=True)
            masks = masks.to(CFG["device"], non_blocking=True)
            out = model(imgs)
            loss = loss_fn(out["out"], masks)
            if "aux" in out:
                loss = loss + 0.4 * loss_fn(out["aux"], masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            trn_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()

        # Validate
        model.eval()
        iou_sum = leaf_sum = n = 0
        with torch.no_grad():
            for imgs, masks, _ in dl_val:
                imgs = imgs.to(CFG["device"])
                masks = masks.to(CFG["device"])
                out = model(imgs)["out"]
                miou, per_cls = iou_score(out, masks, CFG["num_classes"])
                iou_sum += miou
                leaf_sum += per_cls.get("leaf", 0.0)
                n += 1

        val_miou = iou_sum / max(n, 1)
        val_leaf = leaf_sum / max(n, 1)
        avg_loss = trn_loss / max(len(dl_trn), 1)
        history.append({"epoch": epoch+1, "loss": avg_loss,
                        "mIoU": val_miou, "leaf_IoU": val_leaf})
        print(f"Epoch {epoch+1:02d}  loss={avg_loss:.3f}  "
              f"mIoU={val_miou:.3f}  leaf_IoU={val_leaf:.3f}  "
              f"lr={sched.get_last_lr()[0]:.2e}")

        if val_miou > best_iou:
            best_iou = val_miou
            torch.save({"model": model.state_dict(), "cfg": CFG,
                        "iou": val_miou, "leaf_iou": val_leaf,
                        "epoch": epoch+1}, ckpt_path)
            print(
                f"  ↳ saved best (mIoU={val_miou:.3f}  leaf_IoU={val_leaf:.3f})")

    print(f"\nDone. Best val mIoU: {best_iou:.3f}")
    import pandas as pd
    pd.DataFrame(history).to_csv(
        CFG["out_dir"] / "training_history.csv", index=False)

    if CFG["export_conf_maps"]:
        export_confidence_maps(model, ds_val,
                               CFG["out_dir"] / "conf_maps")


# %% Cell 9 — Confidence map export
# After training, exports a heat-map per validation image showing WHERE the
# model is uncertain (mid-grey = ~50% probability = label these images next).
def export_confidence_maps(model, dataset, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    tf_val = build_transforms(CFG["img_size"], train=False)
    print(f"\nExporting confidence maps → {out_dir}")
    for idx in range(len(dataset)):
        _, _, fname = dataset[idx]
        img_path = CFG["data_root"] / "images" / fname
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        H, W = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp = tf_val(image=rgb, mask=np.zeros((H, W), np.uint8))["image"]
        inp = inp.unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(inp)["out"], dim=1)[0, 1].cpu().numpy()
        prob_r = cv2.resize(prob, (W, H))
        heat = cv2.applyColorMap(
            (prob_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
        disp = np.hstack([img_bgr, heat])
        cv2.imwrite(str(out_dir / (Path(fname).stem + "_conf.png")), disp)
    print(f"  Wrote {len(dataset)} confidence maps.")


# %% Cell 10 — Inference helper (importable by infer_and_analyse.py)
def predict_leaf_mask(img_bgr, ckpt_path="models/leaf_segmenter_best.pt",
                      img_size=512, return_prob=False):
    """
    Returns uint8 (0/255) binary leaf mask of the same HxW as the input.
    If return_prob=True, also returns the float32 [0,1] probability map.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tf = build_transforms(img_size, train=False)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    inp = tf(image=rgb, mask=np.zeros((H, W), np.uint8))[
        "image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(inp)["out"], dim=1)[0, 1].cpu().numpy()
        pred = (prob > 0.5).astype(np.uint8)
    mask = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST) * 255
    if return_prob:
        return mask, cv2.resize(prob, (W, H))
    return mask


# %% Cell 11 — Run
if __name__ == "__main__":
    train()
