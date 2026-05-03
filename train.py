"""
train.py
════════════════════════════════════════════════════════════════════════════════
DeepfakeDetector — Training & Validation Script
ResNeXt50_32x4d + LSTM Architecture

Authors : Team DeepScan
Dataset : FaceForensics++ / Custom Collected Dataset
Task    : Binary Classification  (0 = REAL, 1 = FAKE)

════════════════════════════════════════════════════════════════════════════════
DATASET FOLDER STRUCTURE
════════════════════════════════════════════════════════════════════════════════

  dataset/
  ├── train/
  │   ├── real/
  │   │   ├── video_001/        ← one folder per video clip
  │   │   │   ├── frame_0000.jpg
  │   │   │   ├── frame_0001.jpg
  │   │   │   └── ...
  │   │   └── video_002/
  │   └── fake/
  │       ├── video_003/
  │       └── ...
  └── val/
      ├── real/
      └── fake/

Each clip folder should contain pre-extracted face-crop images (JPEG/PNG).
Use any face-extraction pipeline (dlib / MTCNN / RetinaFace) to prepare them.

════════════════════════════════════════════════════════════════════════════════
USAGE EXAMPLES
════════════════════════════════════════════════════════════════════════════════

  # Train with 150 frames (default — matches production model)
  python train.py --data dataset/

  # Train a lightweight 40-frame variant
  python train.py --data dataset/ --frames 40 --epochs 20 --tag model_40f

  # Train with backbone frozen for first 5 epochs
  python train.py --data dataset/ --frames 150 --freeze-epochs 5

  # Resume interrupted training
  python train.py --data dataset/ --frames 150 --resume models/run1/best.pt

  # Full custom configuration
  python train.py \\
      --data          dataset/     \\
      --frames        150          \\
      --hidden        2048         \\
      --dropout       0.4          \\
      --epochs        25           \\
      --batch         8            \\
      --lr            1e-4         \\
      --wd            1e-5         \\
      --freeze-epochs 5            \\
      --patience      7            \\
      --scheduler     cosine       \\
      --tag           model_150f   \\
      --out           models/

════════════════════════════════════════════════════════════════════════════════
OUTPUT
════════════════════════════════════════════════════════════════════════════════

  models/
  └── <tag>/
      ├── best.pt                       ← best validation accuracy
      ├── last.pt                       ← final epoch weights
      ├── model_acc<XX>_f<N>_h<H>.pt   ← descriptively named final model
      └── log.csv                       ← epoch-by-epoch training metrics
"""

import os
import re
import csv
import sys
import time
import math
import random
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn            as nn
import torch.optim         as optim
from torch.utils.data      import Dataset, DataLoader
from torchvision           import transforms
from PIL                   import Image

# ── Local model definition ────────────────────────────────────────────────────
from model import build_model, DeepfakeDetector


# ════════════════════════════════════════════════════════════════════════════
# 0.  Reproducibility
# ════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ════════════════════════════════════════════════════════════════════════════
# 1.  Dataset
# ════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_TO_IDX = {"real": 0, "fake": 1}
IDX_TO_CLASS = {0: "REAL", 1: "FAKE"}


def _collect_clips(root: Path):
    """
    Walk  root/real/  and  root/fake/  and return a list of
    (clip_folder_path, label_int) tuples.

    Supports two layouts:
      • Nested  : root/real/video_001/*.jpg   (one folder = one clip)
      • Flat    : root/real/*.jpg             (all images = one clip)
    """
    clips = []
    for cls_name, label in CLASS_TO_IDX.items():
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            warnings.warn(f"Missing class folder: {cls_dir}")
            continue

        sub_dirs = sorted(p for p in cls_dir.iterdir() if p.is_dir())

        if sub_dirs:
            # Nested layout — each sub-dir is a separate video clip
            for sd in sub_dirs:
                clips.append((sd, label))
        else:
            # Flat layout — treat entire folder as one clip
            clips.append((cls_dir, label))

    return clips


def _load_frames(clip_dir: Path, num_frames: int):
    """
    Uniformly sample `num_frames` images from `clip_dir`.
    Returns a list of RGB numpy arrays (H, W, 3).
    Returns an empty list if the folder has fewer images than needed.
    """
    files = sorted(
        p for p in clip_dir.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    )

    if len(files) < num_frames:
        return []

    indices  = np.linspace(0, len(files) - 1, num_frames, dtype=int)
    selected = [files[i] for i in indices]

    frames = []
    for fp in selected:
        img = cv2.imread(str(fp))
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return frames


class DeepfakeDataset(Dataset):
    """
    Video-level deepfake detection dataset.

    Each item is a tensor of shape  (T, 3, H, W)  where T = num_frames,
    paired with a binary label  (0 = REAL,  1 = FAKE).

    Args
    ────
    root        Path to split directory (contains real/ and fake/ sub-dirs).
    num_frames  Sequence length — must match the model's `frames` argument.
    face_size   Spatial resolution of each face crop (pixels).
    augment     Apply random augmentation (True for train, False for val).
    """

    def __init__(
        self,
        root:       str,
        num_frames: int  = 150,
        face_size:  int  = 112,
        augment:    bool = False,
    ):
        self.root       = Path(root)
        self.num_frames = num_frames
        self.face_size  = face_size
        self.clips      = _collect_clips(self.root)

        if not self.clips:
            raise RuntimeError(
                f"No video clips found under '{root}'.\n"
                "Expected structure:\n"
                "  <root>/real/<clip_name>/*.jpg\n"
                "  <root>/fake/<clip_name>/*.jpg"
            )

        # ── Transforms ────────────────────────────────────────────────────────
        tf = []
        if augment:
            tf += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3,
                    saturation=0.2, hue=0.05
                ),
                transforms.RandomGrayscale(p=0.02),
            ]
        tf += [
            transforms.Resize((face_size, face_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform = transforms.Compose(tf)

    # ── Internals ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip_dir, label = self.clips[idx]
        frames = _load_frames(clip_dir, self.num_frames)

        # Fallback: pick a random clip if this one is too short
        retries = 0
        while len(frames) < self.num_frames and retries < 10:
            alt = random.randint(0, len(self.clips) - 1)
            clip_dir, label = self.clips[alt]
            frames = _load_frames(clip_dir, self.num_frames)
            retries += 1

        # Last resort: repeat last frame to fill sequence
        if frames:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        else:
            # Return zeros if completely empty (edge case)
            dummy = torch.zeros(self.num_frames, 3,
                                self.face_size, self.face_size)
            return dummy, torch.tensor(label, dtype=torch.long)

        tensors = [
            self.transform(Image.fromarray(f))
            for f in frames[:self.num_frames]
        ]
        seq = torch.stack(tensors, dim=0)          # (T, 3, H, W)
        return seq, torch.tensor(label, dtype=torch.long)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def label_counts(self):
        counts = {0: 0, 1: 0}
        for _, l in self.clips:
            counts[l] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for imbalanced datasets."""
        counts = self.label_counts()
        total  = sum(counts.values())
        w = torch.tensor(
            [total / (2 * max(counts[0], 1)),
             total / (2 * max(counts[1], 1))],
            dtype=torch.float32
        )
        return w


# ════════════════════════════════════════════════════════════════════════════
# 2.  Training utilities
# ════════════════════════════════════════════════════════════════════════════

class AverageMeter:
    """Computes and stores a running mean."""
    def __init__(self):  self.reset()
    def reset(self):     self.sum = self.count = 0.0
    @property
    def avg(self):       return self.sum / self.count if self.count else 0.0
    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ── One training epoch ────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, scaler,
                epoch, total_epochs):
    """
    Train for one full epoch.
    Returns (avg_loss, avg_accuracy).
    """
    model.train()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    n_batches = len(loader)

    for step, (seqs, labels) in enumerate(loader, 1):
        seqs   = seqs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward ───────────────────────────────────────────────────────────
        if scaler is not None:                              # AMP (GPU)
            with torch.cuda.amp.autocast():
                logits = model(seqs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:                                               # FP32 (CPU / GPU)
            logits = model(seqs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs   = seqs.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(accuracy(logits, labels), bs)

        # ── Inline progress bar ───────────────────────────────────────────────
        pct = step / n_batches * 100
        bar = "█" * int(pct / 4) + "░" * (25 - int(pct / 4))
        print(
            f"\r  Ep {epoch:>3}/{total_epochs}  [{bar}] {pct:5.1f}%  "
            f"loss={loss_m.avg:.4f}  acc={acc_m.avg * 100:6.2f}%",
            end="", flush=True,
        )

    print()   # newline after progress
    return loss_m.avg, acc_m.avg


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluate on a validation split.
    Returns (avg_loss, avg_accuracy, per_class_accuracy_dict).
    """
    model.eval()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    # Track per-class correct / total for detailed reporting
    per_class_correct = {0: 0, 1: 0}
    per_class_total   = {0: 0, 1: 0}

    for seqs, labels in loader:
        seqs   = seqs.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(seqs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        bs = seqs.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(accuracy(logits, labels), bs)

        for c in [0, 1]:
            mask = labels == c
            per_class_correct[c] += (preds[mask] == c).sum().item()
            per_class_total[c]   += mask.sum().item()

    per_class_acc = {
        IDX_TO_CLASS[c]: (
            per_class_correct[c] / per_class_total[c]
            if per_class_total[c] > 0 else 0.0
        )
        for c in [0, 1]
    }

    return loss_m.avg, acc_m.avg, per_class_acc


# ════════════════════════════════════════════════════════════════════════════
# 3.  Checkpoint helpers
# ════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def resume_from(path: str, model: nn.Module, device: torch.device) -> None:
    sd = torch.load(path, map_location=device)
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"  ✓  Resumed from  {path}")


# ════════════════════════════════════════════════════════════════════════════
# 4.  Backbone freeze / unfreeze
# ════════════════════════════════════════════════════════════════════════════

def freeze_backbone(model: DeepfakeDetector) -> None:
    for p in model.model.parameters():
        p.requires_grad = False
    print("  ── Backbone FROZEN  (only LSTM + classifier will train)")


def unfreeze_backbone(model: DeepfakeDetector) -> None:
    for p in model.model.parameters():
        p.requires_grad = True
    print("  ── Backbone UNFROZEN  (full end-to-end training)")


# ════════════════════════════════════════════════════════════════════════════
# 5.  Argument parser
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="DeepfakeDetector — Training & Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--data",      required=True,
                   help="Root dataset dir  (must contain train/ and val/)")
    g.add_argument("--workers",   type=int,   default=2,
                   help="DataLoader worker processes")
    g.add_argument("--face-size", type=int,   default=112,
                   help="Face crop resolution (px)")

    # Architecture
    g = p.add_argument_group("Architecture")
    g.add_argument("--frames",    type=int,   default=150,
                   help="Frames per clip  (sequence length T)")
    g.add_argument("--hidden",    type=int,   default=2048,
                   help="LSTM hidden units")
    g.add_argument("--classes",   type=int,   default=2,
                   help="Output classes")
    g.add_argument("--dropout",   type=float, default=0.4,
                   help="Dropout before classifier")

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",         type=int,   default=25)
    g.add_argument("--batch",          type=int,   default=4)
    g.add_argument("--lr",             type=float, default=1e-4,
                   help="Initial learning rate")
    g.add_argument("--wd",             type=float, default=1e-5,
                   help="AdamW weight decay")
    g.add_argument("--freeze-epochs",  type=int,   default=0,
                   help="Freeze ResNeXt backbone for first N epochs")
    g.add_argument("--patience",       type=int,   default=7,
                   help="Early-stop patience (epochs without val improvement)")
    g.add_argument("--scheduler",
                   choices=["cosine", "step", "none"], default="cosine")
    g.add_argument("--lr-step",        type=int,   default=10,
                   help="StepLR: decay every N epochs")
    g.add_argument("--lr-gamma",       type=float, default=0.5,
                   help="StepLR decay factor")
    g.add_argument("--weighted-loss",  action="store_true",
                   help="Use class-frequency-weighted cross-entropy")
    g.add_argument("--amp",            action="store_true",
                   help="Mixed-precision training  (GPU only)")
    g.add_argument("--seed",           type=int,   default=42)

    # Resume / output
    g = p.add_argument_group("Output")
    g.add_argument("--resume",    type=str, default=None,
                   help="Path to .pt checkpoint to resume from")
    g.add_argument("--out",       type=str, default="models",
                   help="Root output directory for checkpoints")
    g.add_argument("--tag",       type=str, default="",
                   help="Run tag (used as output sub-folder prefix)")

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    use_amp = args.amp and device.type == "cuda"

    # ── Output directory ──────────────────────────────────────────────────────
    run_id  = f"f{args.frames}_h{args.hidden}_e{args.epochs}"
    run_id  = f"{args.tag}_{run_id}" if args.tag else run_id
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = str(run_dir / "best.pt")
    last_path = str(run_dir / "last.pt")
    log_path  = str(run_dir / "log.csv")

    # ── Banner ────────────────────────────────────────────────────────────────
    SEP  = "═" * 68
    sep2 = "─" * 68
    print(f"\n{SEP}")
    print(f"  DeepScan — Deepfake Detection Model Training")
    print(sep2)
    print(f"  Run ID    : {run_id}")
    print(f"  Device    : {device}"
          + ("  (AMP enabled)" if use_amp else ""))
    print(f"  Frames    : {args.frames}   Hidden : {args.hidden}"
          f"   Dropout : {args.dropout}")
    print(f"  Epochs    : {args.epochs}   Batch  : {args.batch}"
          f"   LR      : {args.lr}")
    print(f"  Scheduler : {args.scheduler}"
          f"   Patience: {args.patience}")
    print(f"  Output    : {run_dir}")
    print(SEP)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print(f"\n  Loading dataset from  '{args.data}' …")

    train_ds = DeepfakeDataset(
        root       = str(Path(args.data) / "train"),
        num_frames = args.frames,
        face_size  = args.face_size,
        augment    = True,
    )
    val_ds = DeepfakeDataset(
        root       = str(Path(args.data) / "val"),
        num_frames = args.frames,
        face_size  = args.face_size,
        augment    = False,
    )

    tr_counts  = train_ds.label_counts()
    val_counts = val_ds.label_counts()

    print(f"\n  {'Split':<8} {'Total':>6}  {'Real':>6}  {'Fake':>6}")
    print(f"  {sep2[:40]}")
    print(f"  {'Train':<8} {len(train_ds):>6}  "
          f"{tr_counts[0]:>6}  {tr_counts[1]:>6}")
    print(f"  {'Val':<8} {len(val_ds):>6}  "
          f"{val_counts[0]:>6}  {val_counts[1]:>6}")

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch,
        shuffle     = True,
        num_workers = args.workers,
        pin_memory  = device.type == "cuda",
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch,
        shuffle     = False,
        num_workers = args.workers,
        pin_memory  = device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n  Building model  (frames={args.frames}, "
          f"hidden={args.hidden}, pretrained backbone) …")

    model = build_model(
        frames     = args.frames,
        hidden     = args.hidden,
        classes    = args.classes,
        dropout    = args.dropout,
        pretrained = True,
    ).to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {total_p:,}  (trainable: {trainable_p:,})")

    if args.resume:
        resume_from(args.resume, model, device)

    # ── Loss function ─────────────────────────────────────────────────────────
    if args.weighted_loss:
        w = train_ds.class_weights().to(device)
        print(f"  Class weights  →  REAL: {w[0]:.3f}   FAKE: {w[1]:.3f}")
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd,
    )

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-7
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step, gamma=args.lr_gamma
        )
    else:
        scheduler = None

    # ── Mixed precision ───────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── CSV logger ────────────────────────────────────────────────────────────
    log_fields = [
        "epoch", "lr",
        "train_loss", "train_acc",
        "val_loss",   "val_acc",
        "real_acc",   "fake_acc",
        "epoch_time_s",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # ═════════════════════════════════════════════════════════════════════════
    # Training loop
    # ═════════════════════════════════════════════════════════════════════════

    print(f"\n{sep2}")
    print(f"  {'Ep':>4}  {'LR':>9}  "
          f"{'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>8}  {'ValAcc':>8}  "
          f"{'REAL%':>7}  {'FAKE%':>7}  {'Time':>6}")
    print(f"  {sep2}")

    best_val_acc  = 0.0
    patience_left = args.patience
    run_start     = time.perf_counter()
    last_epoch    = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        ep_t0 = time.perf_counter()

        # ── Backbone freeze / unfreeze schedule ───────────────────────────────
        if epoch == 1 and args.freeze_epochs > 0:
            freeze_backbone(model)

        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            unfreeze_backbone(model)
            # Rebuild optimizer to include newly unfrozen backbone params
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.wd
            )

        # ── Train ─────────────────────────────────────────────────────────────
        t_loss, t_acc = train_epoch(
            model, train_loader, criterion,
            optimizer, device, scaler,
            epoch, args.epochs,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        v_loss, v_acc, per_cls = validate(
            model, val_loader, criterion, device
        )

        # ── Step scheduler ────────────────────────────────────────────────────
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        ep_time = time.perf_counter() - ep_t0

        # ── Best model checkpoint ─────────────────────────────────────────────
        is_best = v_acc > best_val_acc
        flag    = " ◀ BEST" if is_best else ""

        if is_best:
            best_val_acc  = v_acc
            patience_left = args.patience
            save_checkpoint(model, best_path)
        else:
            patience_left -= 1

        # ── Print row ─────────────────────────────────────────────────────────
        print(
            f"  {epoch:>4}  {current_lr:>9.2e}  "
            f"{t_loss:>10.4f}  {t_acc*100:>8.2f}%  "
            f"{v_loss:>8.4f}  {v_acc*100:>7.2f}%  "
            f"{per_cls['REAL']*100:>6.1f}%  "
            f"{per_cls['FAKE']*100:>6.1f}%  "
            f"{ep_time:>5.0f}s"
            f"{flag}"
        )

        # ── CSV row ───────────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch":       epoch,
                "lr":          round(current_lr, 8),
                "train_loss":  round(t_loss, 6),
                "train_acc":   round(t_acc * 100, 4),
                "val_loss":    round(v_loss, 6),
                "val_acc":     round(v_acc * 100, 4),
                "real_acc":    round(per_cls["REAL"] * 100, 4),
                "fake_acc":    round(per_cls["FAKE"] * 100, 4),
                "epoch_time_s": round(ep_time, 1),
            })

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_left <= 0:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # ═════════════════════════════════════════════════════════════════════════
    # Post-training
    # ═════════════════════════════════════════════════════════════════════════

    save_checkpoint(model, last_path)

    # Descriptive final model filename
    acc_tag    = int(best_val_acc * 100)
    final_name = (
        f"model_acc{acc_tag}"
        f"_f{args.frames}"
        f"_h{args.hidden}"
        f"_e{last_epoch}.pt"
    )
    final_path = str(run_dir / final_name)
    save_checkpoint(model, final_path)

    total_time = time.perf_counter() - run_start

    print(f"\n{SEP}")
    print(f"  Training complete")
    print(sep2)
    print(f"  Best val accuracy   : {best_val_acc * 100:.2f}%")
    print(f"  Total training time : {total_time / 60:.1f} min "
          f"({total_time:.0f}s)")
    print(sep2)
    print(f"  Checkpoints saved:")
    print(f"    Best   →  {best_path}")
    print(f"    Last   →  {last_path}")
    print(f"    Final  →  {final_path}")
    print(f"    Log    →  {log_path}")
    print(sep2)
    print(f"  To use in the app:")
    print(f"    1. Copy  {final_path}  →  deepfake_app/models/")
    print(f"    2. The model was trained with --frames {args.frames}")
    print(f"       Make sure SEQUENCE_LENGTH = {args.frames} in pipeline.py")
    print(SEP + "\n")

    # ── Final validation report ───────────────────────────────────────────────
    print(f"  Loading best checkpoint for final validation report …")
    best_sd = torch.load(best_path, map_location=device)
    model.load_state_dict(best_sd, strict=False)

    final_loss, final_acc, final_cls = validate(
        model, val_loader, criterion, device
    )

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │        FINAL VALIDATION REPORT          │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Overall accuracy  : {final_acc*100:>6.2f}%           │")
    print(f"  │  Loss              : {final_loss:>8.4f}             │")
    print(f"  │  REAL accuracy     : {final_cls['REAL']*100:>6.2f}%           │")
    print(f"  │  FAKE accuracy     : {final_cls['FAKE']*100:>6.2f}%           │")
    print(f"  │  Frames used       : {args.frames:<6}               │")
    print(f"  │  LSTM hidden       : {args.hidden:<6}               │")
    print(f"  └─────────────────────────────────────────┘\n")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()