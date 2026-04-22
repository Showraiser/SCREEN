"""
Step 2 — Encode frames using OpenAI CLIP (ViT-B/32).
Saves per-folder .npy files containing L2-normalised 512-D feature vectors.
"""

import os
import re
import argparse
import numpy as np
import torch
import clip
from PIL import Image


def extract_number(filename):
    numbers = re.findall(r"\d+", filename)
    return int(numbers[-1]) if numbers else -1


def encode_folder(dirpath, output_dir, model, preprocess, device):
    image_files = sorted(
        [f for f in os.listdir(dirpath) if f.lower().endswith((".jpg", ".png"))],
        key=extract_number,
    )
    if not image_files:
        return

    os.makedirs(output_dir, exist_ok=True)
    features, filenames = [], []

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(dirpath, img_name)
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  [SKIP] {img_name}: {e}")
            continue

        with torch.no_grad():
            feat = model.encode_image(image)
            feat /= feat.norm(dim=-1, keepdim=True)

        features.append(feat.cpu().numpy())
        filenames.append(img_name)
        print(f"  [{idx + 1}/{len(image_files)}] {img_name}")

    if not features:
        print("  No valid frames found — skipping.")
        return

    features = np.vstack(features)
    out_path = os.path.join(output_dir, "clip_features.npy")
    np.save(out_path, features)

    # Save matching filenames so index building can reconstruct paths exactly
    names_path = os.path.join(output_dir, "frame_names.npy")
    np.save(names_path, np.array(filenames))

    print(f"  Saved {features.shape[0]} vectors → {out_path}")


def run(base_dir, output_root, model, preprocess, device):
    for dirpath, dirs, _ in os.walk(base_dir):
        # Only process leaf directories (those containing images)
        rel_path = os.path.relpath(dirpath, base_dir)
        output_dir = os.path.join(output_root, rel_path)
        print(f"\nProcessing: {rel_path}")
        encode_folder(dirpath, output_dir, model, preprocess, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP features from video frames.")
    parser.add_argument("--frames_dir", default="frames", help="Root folder of extracted frames (default: ./frames).")
    parser.add_argument("--output_dir", default="clip_features", help="Root folder to save .npy features (default: ./clip_features).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    os.makedirs(args.output_dir, exist_ok=True)
    run(args.frames_dir, args.output_dir, model, preprocess, device)
    print("\nDone.")
