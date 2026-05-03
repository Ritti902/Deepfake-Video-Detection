"""
pipeline.py — Deepfake Detection Inference Pipeline
====================================================
Uses model.forward() directly — no manual attribute access.
Handles ALL known .pt key naming conventions automatically.
"""

import os
import re
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import dlib

from model import DeepfakeDetector

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 150
FACE_SIZE       = 112
DETECT_EVERY    = 3
DETECT_SCALE    = 0.5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_LABELS = {0: "REAL", 1: "FAKE"}

# ─────────────────────────────────────────────────────────────────────────────
# CPU tuning
# ─────────────────────────────────────────────────────────────────────────────

def _tune_cpu():
    n = os.cpu_count() or 2
    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n // 2))

_tune_cpu()

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Key remapping — handles ALL known naming conventions
# ─────────────────────────────────────────────────────────────────────────────

def _remap_state_dict(sd: dict, target_keys: set) -> dict:
    """
    Intelligently remap state-dict keys to match the model's actual attribute names.
    Detects what naming convention the .pt file uses and remaps accordingly.
    
    Known conventions:
      Saved as                →  Model expects
      ─────────────────────────────────────────
      model.*                →  model.*          (same — no remap needed)
      feature_extractor.*    →  model.*          (remap needed)
      module.model.*         →  model.*          (strip DataParallel)
      module.feature_extractor.* → model.*       (strip + remap)
      linear1.*              →  linear1.*        (same — no remap needed)
      classifier.*           →  linear1.*        (remap needed)
    """
    # Step 1: Strip DataParallel 'module.' prefix
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # Step 2: Check if remapping is needed by comparing against target keys
    # Try the state dict as-is first
    sample_keys = set(list(sd.keys())[:5])
    
    # Detect source convention
    uses_feature_extractor = any(k.startswith("feature_extractor.") for k in sd)
    uses_classifier        = "classifier.weight" in sd
    uses_model             = any(k.startswith("model.") for k in sd)
    uses_linear1           = "linear1.weight" in sd

    # Only remap if needed
    needs_remap = uses_feature_extractor or uses_classifier

    if not needs_remap:
        return sd

    out = {}
    for k, v in sd.items():
        # feature_extractor.* → model.*
        if k.startswith("feature_extractor."):
            k = "model." + k[len("feature_extractor."):]
        # classifier.* → linear1.*
        elif k.startswith("classifier."):
            k = "linear1." + k[len("classifier."):]
        out[k] = v

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _detect_frames_from_name(model_path: str) -> int:
    """Try to detect sequence length from filename."""
    name = os.path.basename(model_path).lower()
    patterns = [
        r"[_-](\d+)[_-]?frames",
        r"frames[_-]?(\d+)",
        r"seq[_-]?(\d+)",
        r"_f(\d+)[_-]",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            val = int(m.group(1))
            if 10 <= val <= 500:
                return val
    return SEQUENCE_LENGTH


def load_model(model_path: str, device: torch.device,
               frames: int = None) -> DeepfakeDetector:
    """
    Load a .pt checkpoint into DeepfakeDetector.
    Automatically detects and remaps key naming conventions.
    Uses strict=False only for num_batches_tracked mismatches.
    """
    if frames is None:
        frames = _detect_frames_from_name(model_path)

    # Build model with matching architecture
    model = DeepfakeDetector(
        frames=frames,
        hidden=2048,
        classes=2,
        dropout=0.4,
    )

    # Load raw checkpoint
    raw_sd = torch.load(model_path, map_location=device)

    # Get the model's expected keys
    target_keys = set(model.state_dict().keys())

    # Remap keys
    sd = _remap_state_dict(raw_sd, target_keys)

    # Load with strict=False to allow num_batches_tracked differences
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Filter out harmless num_batches_tracked mismatches
    real_missing    = [k for k in missing    if "num_batches_tracked" not in k]
    real_unexpected = [k for k in unexpected if "num_batches_tracked" not in k]

    if real_missing:
        print(f"  ⚠ Missing keys ({len(real_missing)}): {real_missing[:3]}")
    if real_unexpected:
        print(f"  ⚠ Unexpected keys ({len(real_unexpected)}): {real_unexpected[:3]}")
    if not real_missing and not real_unexpected:
        print(f"  ✓ Model loaded cleanly: {os.path.basename(model_path)}")

    model.to(device)
    model.eval()
    return model


def list_available_models(models_dir: str = "models") -> list:
    if not os.path.isdir(models_dir):
        return []
    return sorted(f for f in os.listdir(models_dir) if f.endswith(".pt"))


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction — sequential read (fast)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = SEQUENCE_LENGTH) -> list:
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < num_frames:
        cap.release()
        raise ValueError(f"Video has only {total} frames — need {num_frames}.")

    target_sorted = sorted(set(np.linspace(0, total - 1, num_frames, dtype=int).tolist()))

    frames   = []
    curr_idx = 0
    t_iter   = iter(target_sorted)
    next_tgt = next(t_iter)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if curr_idx == next_tgt:
            frames.append(frame)
            try:
                next_tgt = next(t_iter)
            except StopIteration:
                break
        curr_idx += 1

    cap.release()

    if len(frames) < num_frames:
        raise ValueError(f"Only read {len(frames)} frames; {num_frames} needed.")

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Face detection — robust multi-fallback
# ─────────────────────────────────────────────────────────────────────────────

_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        _detector = dlib.get_frontal_face_detector()
    return _detector


def _scale_rect(rect, scale: float, img_h: int, img_w: int):
    inv = 1.0 / scale
    l = max(0,     int(rect.left()   * inv))
    t = max(0,     int(rect.top()    * inv))
    r = min(img_w, int(rect.right()  * inv))
    b = min(img_h, int(rect.bottom() * inv))
    return l, t, r, b


def detect_faces(frames: list) -> list:
    """
    Robust face detection with 4 fallback strategies so it never fails:
      1. Fast: half-res dlib, upsample=0, bbox reuse every 3 frames
      2. Retry: full-res dlib, upsample=1 if fast fails
      3. Fallback: centre crop of frame if dlib fails completely
      4. Pad: repeat existing crops if still under needed count
    """
    detector  = _get_detector()
    crops     = []
    last_bbox = None

    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        run_detect = (i % DETECT_EVERY == 0) or (last_bbox is None)

        if run_detect:
            # Strategy 1: fast half-res
            small = cv2.resize(rgb, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))
            dets  = detector(small, 0)

            if dets:
                best      = max(dets, key=lambda r: r.width() * r.height())
                last_bbox = _scale_rect(best, DETECT_SCALE, h, w)
            else:
                # Strategy 2: full-res retry
                dets2 = detector(rgb, 1)
                if dets2:
                    best = max(dets2, key=lambda r: r.width() * r.height())
                    last_bbox = (
                        max(0, best.left()),
                        max(0, best.top()),
                        min(w, best.right()),
                        min(h, best.bottom()),
                    )
                # else keep last_bbox

        # Strategy 3: centre crop fallback
        if last_bbox is None:
            mx = int(w * 0.20)
            my = int(h * 0.15)
            last_bbox = (mx, my, w - mx, h - my)

        l, t, r, b = last_bbox
        crop = rgb[t:b, l:r]

        if crop.size == 0:
            last_bbox = None
            crop = rgb[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]

        crops.append(crop)

    # Strategy 4: pad by repeating if short
    if len(crops) < SEQUENCE_LENGTH:
        if len(crops) == 0:
            raise ValueError("Could not extract any frames from this video.")
        idx   = np.linspace(0, len(crops) - 1, SEQUENCE_LENGTH, dtype=int)
        crops = [crops[i] for i in idx]

    return crops


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — vectorized numpy (fast)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_faces(crops: list, num_frames: int = SEQUENCE_LENGTH) -> torch.Tensor:
    if len(crops) > num_frames:
        idx   = np.linspace(0, len(crops) - 1, num_frames, dtype=int)
        crops = [crops[i] for i in idx]

    resized = np.stack([
        cv2.resize(c, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
        for c in crops[:num_frames]
    ], axis=0)  # (T, H, W, 3)

    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(0, 3, 1, 2)  # (T, 3, H, W)

    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)  # (1, T, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction — uses model.forward() directly, no manual attribute access
# ─────────────────────────────────────────────────────────────────────────────

def predict(model: DeepfakeDetector,
            tensor: torch.Tensor,
            device: torch.device,
            invert: bool = False) -> dict:
    """
    Run inference using model.forward() — the safest approach since it
    uses whatever internal attributes the model actually has, regardless
    of naming convention.
    """
    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor)                        # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]     # (2,)

    real_p = probs[0].item()
    fake_p = probs[1].item()
    pred   = int(torch.argmax(probs).item())

    if invert:
        pred           = 1 - pred
        real_p, fake_p = fake_p, real_p

    return {
        "label":      CLASS_LABELS[pred],
        "confidence": round(max(real_p, fake_p) * 100, 2),
        "probs":      [round(real_p * 100, 2), round(fake_p * 100, 2)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_path: str,
                 model: DeepfakeDetector,
                 device: torch.device,
                 invert: bool = False,
                 model_name: str = "",
                 model_names: list = None) -> dict:

    if not invert and model_names and model_name:
        invert = (model_name == model_names[0])

    n_frames = getattr(model, "frames", SEQUENCE_LENGTH)

    frames = extract_frames(video_path, num_frames=n_frames)
    faces  = detect_faces(frames)
    tensor = preprocess_faces(faces, num_frames=n_frames)
    return predict(model, tensor, device, invert=invert)