"""
Run this script on your Mac BEFORE uploading to Kaggle.
It will create a folder called 'hospital_adapters_for_kaggle' on your Desktop,
organized exactly as Kaggle expects it.
After running, upload that folder to Kaggle as a Dataset named 'hospital-adapters'.
"""

import os
import shutil
from pathlib import Path

# ── Source folders on your Mac ──────────────────────────────────────────────
QWEN_SOURCE = Path("/Users/Genai Project/finetuning models/qwen/content/qwen2_5_1_5b_instruct_adapter")
MISTRAL_SOURCE = Path("/Users/Genai Project/finetuning models/mistral/mistral_v02_adapter_fixed_for_runtime")

# ── Destination: organized folder on your Desktop ───────────────────────────
DEST_ROOT = Path.home() / "Desktop" / "hospital_adapters_for_kaggle"
QWEN_DEST    = DEST_ROOT / "qwen_adapter"
MISTRAL_DEST = DEST_ROOT / "mistral_adapter"

# ── Files to copy for each model ────────────────────────────────────────────
QWEN_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
]

MISTRAL_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
]

def copy_files(source: Path, dest: Path, filenames: list, model_name: str):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"  Copying {model_name} adapter files...")
    print(f"{'='*55}")
    for fname in filenames:
        src_file = source / fname
        dst_file = dest / fname
        if not src_file.exists():
            print(f"  ⚠️  MISSING: {fname}  ← check source folder")
            continue
        size_mb = src_file.stat().st_size / 1e6
        print(f"  Copying {fname}  ({size_mb:.1f} MB) ...", end=" ", flush=True)
        shutil.copy2(src_file, dst_file)
        print("done ✅")

def main():
    print("\n" + "="*55)
    print("  Hospital Adapters — Kaggle Upload Organizer")
    print("="*55)

    # Verify source folders exist
    if not QWEN_SOURCE.exists():
        print(f"\n❌ ERROR: Qwen source folder not found:\n   {QWEN_SOURCE}")
        return
    if not MISTRAL_SOURCE.exists():
        print(f"\n❌ ERROR: Mistral source folder not found:\n   {MISTRAL_SOURCE}")
        return

    # Clean destination if it already exists
    if DEST_ROOT.exists():
        print(f"\nRemoving old destination folder...")
        shutil.rmtree(DEST_ROOT)

    # Copy Qwen files
    copy_files(QWEN_SOURCE, QWEN_DEST, QWEN_FILES, "Qwen")

    # Copy Mistral files
    copy_files(MISTRAL_SOURCE, MISTRAL_DEST, MISTRAL_FILES, "Mistral")

    # Print final structure
    print(f"\n{'='*55}")
    print("  DONE. Final folder structure:")
    print(f"{'='*55}")
    for root, dirs, files in os.walk(DEST_ROOT):
        level = str(root).replace(str(DEST_ROOT), "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            size_mb = (Path(root) / f).stat().st_size / 1e6
            print(f"{indent}  {f}  ({size_mb:.1f} MB)")

    print(f"\n{'='*55}")
    print(f"  Folder saved to your Desktop:")
    print(f"  {DEST_ROOT}")
    print(f"{'='*55}")
    print("\n  NEXT STEP:")
    print("  1. Go to kaggle.com → Create → New Dataset")
    print("  2. Name it exactly:  hospital-adapters")
    print("  3. Upload the folder:  hospital_adapters_for_kaggle/")
    print("     (drag and drop the whole folder)")
    print("  4. Set visibility: Private → Create\n")

if __name__ == "__main__":
    main()
