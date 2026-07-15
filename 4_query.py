"""
Step 4 — Query SCREEN: identify a movie from a short video clip.

Usage:
    python 4_query.py                             # interactive mode (default)
    python 4_query.py --clip path/to/clip.mp4     # single-shot mode
"""

import os
import glob
import shutil
import argparse
import subprocess
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
import clip
import faiss
import joblib
from PIL import Image


def extract_frames(video_path, frames_dir, fps=1):
    os.makedirs(frames_dir, exist_ok=True)
    command = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(frames_dir, "frame_%03d.jpg"),
        "-hide_banner", "-loglevel", "error",
    ]
    result = subprocess.run(command)
    return result.returncode == 0


def encode_frames(frames_dir, model, preprocess, device):
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not frame_paths:
        return None, 0

    features = []
    for path in frame_paths:
        try:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
        except Exception as e:
            print(f"  [SKIP] {path}: {e}")
            continue
        with torch.no_grad():
            feat = model.encode_image(image)
            feat /= feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy())

    if not features:
        return None, 0
    return np.vstack(features).astype("float32"), len(frame_paths)


def query_clip(video_path, frames_dir, model, preprocess, device, pca, index, metadata):
    print(f"\nProcessing: {video_path}")

    # Clean temp frames
    for f in glob.glob(os.path.join(frames_dir, "*.jpg")):
        os.remove(f)

    if not extract_frames(video_path, frames_dir):
        print("  FFmpeg extraction failed.")
        return

    features, n_frames = encode_frames(frames_dir, model, preprocess, device)
    if features is None:
        print("  No frames could be encoded.")
        return

    print(f"  {n_frames} frames extracted and encoded.")

    features_pca = pca.transform(features).astype("float32")
    _, I = index.search(features_pca, k=1)

    matched = [
        metadata.iloc[idx][["movie_name", "year", "director", "genre"]]
        for idx in I[:, 0]
    ]
    counts = Counter(tuple(m.items()) for m in matched)
    best, match_count = counts.most_common(1)[0]
    best = dict(best)

    print("\n── SCREEN Result ──────────────────────────────")
    print(f"  Movie    : {best.get('movie_name')}")
    print(f"  Year     : {best.get('year')}")
    print(f"  Director : {best.get('director')}")
    print(f"  Genre    : {best.get('genre')}")
    print(f"  Confidence: {match_count}/{n_frames} frames matched")
    print("────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(description="SCREEN — identify a movie from a short clip.")
    parser.add_argument("--index_dir", default="index", help="Folder containing index artefacts (default: ./index).")
    parser.add_argument("--clip", default=None, help="Path to a single query clip. Omit for interactive mode.")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to sample from the query clip (default: 1).")
    args = parser.parse_args()

    index_path = os.path.join(args.index_dir, "screen_hnsw.index")
    metadata_csv = os.path.join(args.index_dir, "metadata.csv")
    pca_path = os.path.join(args.index_dir, "pca_model.joblib")

    for p in [index_path, metadata_csv, pca_path]:
        if not os.path.exists(p):
            print(f"Missing artefact: {p}  — run step 3 first.")
            return

    print("Loading models…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    index = faiss.read_index(index_path)
    metadata = pd.read_csv(metadata_csv)
    pca = joblib.load(pca_path)
    print(f"Index ready — {index.ntotal} vectors, device: {device}\n")

    frames_dir = tempfile.mkdtemp(prefix="screen_query_")
    try:
        if args.clip:
            # Single-shot mode
            if not os.path.exists(args.clip):
                print(f"File not found: {args.clip}")
                return
            query_clip(args.clip, frames_dir, model, preprocess, device, pca, index, metadata)
        else:
            # Interactive mode
            print("Interactive mode — paste a clip path or type 'q' to quit.\n")
            while True:
                video_path = input("Clip path: ").strip().strip('"').strip("'")
                if video_path.lower() == "q":
                    print("Exiting.")
                    break
                if not os.path.exists(video_path):
                    print("  File not found. Try again.")
                    continue
                query_clip(video_path, frames_dir, model, preprocess, device, pca, index, metadata)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
