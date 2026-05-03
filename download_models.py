"""
download_models.py
──────────────────
Downloads all model (.pt) files from the shared Google Drive folder
into the local  models/  directory.

Usage:
    python download_models.py

Requirements:
    pip install gdown
"""

import os
import sys

# ── Google Drive folder ID ────────────────────────────────────────────────────
# Extracted from:
# https://drive.google.com/drive/folders/1P-Ze64c1JMZTvg9ve0w5G8mUMgEkrIDN
FOLDER_ID   = "1P-Ze64c1JMZTvg9ve0w5G8mUMgEkrIDN"
MODELS_DIR  = "models"
# ─────────────────────────────────────────────────────────────────────────────


def check_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[download_models] 'gdown' not found. Installing …")
        os.system(f"{sys.executable} -m pip install gdown --quiet")


def main():
    check_gdown()
    import gdown

    os.makedirs(MODELS_DIR, exist_ok=True)

    folder_url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    print(f"[download_models] Downloading models from:\n  {folder_url}")
    print(f"[download_models] Saving to: ./{MODELS_DIR}/\n")

    gdown.download_folder(
        url=folder_url,
        output=MODELS_DIR,
        quiet=False,
        use_cookies=False,
    )

    pts = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]
    if pts:
        print(f"\n✅ Downloaded {len(pts)} model(s):")
        for f in pts:
            size_mb = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1e6
            print(f"   • {f}  ({size_mb:.1f} MB)")
    else:
        print("\n⚠️  No .pt files found after download.")
        print("    Check that the Drive folder is publicly accessible.")


if __name__ == "__main__":
    main()