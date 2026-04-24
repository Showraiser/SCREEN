"""
SCREEN inference engine — loaded once at startup, shared across requests.
"""

import glob
import io
import os
import shutil
import subprocess
import tempfile

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image


class SCREENInference:
    def __init__(self, index_dir: str = "index"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SCREEN] Loading CLIP on {self.device}…")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        index_path = os.path.join(index_dir, "screen_hnsw.index")
        pca_path   = os.path.join(index_dir, "pca_model.joblib")
        meta_path  = os.path.join(index_dir, "metadata.csv")

        for p in [index_path, pca_path, meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing artefact: {p}. Run steps 1-3 first."
                )

        print("[SCREEN] Loading FAISS index…")
        self.index    = faiss.read_index(index_path)
        self.pca      = joblib.load(pca_path)
        self.metadata = pd.read_csv(meta_path)
        print(f"[SCREEN] Ready — {self.index.ntotal} vectors indexed.")

    # ── internal ──────────────────────────────────────────────────────────────
    def _encode_pil(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype("float32")

    def _search(self, vectors: np.ndarray, top_k: int) -> list[dict]:
        """PCA → FAISS search over a batch of CLIP vectors. Aggregates by movie."""
        feat_pca = self.pca.transform(vectors).astype("float32")
        D, I     = self.index.search(feat_pca, k=top_k)

        scores:     dict[str, float] = {}
        meta_cache: dict[str, dict]  = {}

        for dists, idxs in zip(D, I):
            for dist, idx in zip(dists, idxs):
                row   = self.metadata.iloc[idx]
                movie = row["movie_name"]
                conf  = float(1.0 / (1.0 + dist))
                scores[movie]     = scores.get(movie, 0.0) + conf
                meta_cache[movie] = {
                    "movie_name": movie,
                    "year":       str(row.get("year", "")),
                    "director":   str(row.get("director", "")),
                    "genre":      str(row.get("genre", "")),
                }

        n_frames = len(D)
        ranked   = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results  = []
        for movie, total_conf in ranked[:top_k]:
            entry = meta_cache[movie].copy()
            entry["confidence"] = round(total_conf / n_frames, 4)
            results.append(entry)

        return results

    # ── public API ────────────────────────────────────────────────────────────
    def query_image(self, image_bytes: bytes, top_k: int = 5) -> list[dict]:
        """Identify from a single image frame."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        feat  = self._encode_pil(image)
        return self._search(feat, top_k)

    def query_video(
        self,
        video_bytes: bytes,
        video_ext:   str = ".mp4",
        fps:         int = 1,
        top_k:       int = 5,
    ) -> tuple[list[dict], int]:
        """
        Identify from a video clip.
        Extracts frames with FFmpeg, encodes each with CLIP, returns results + frame count.
        """
        tmp_dir = tempfile.mkdtemp(prefix="screen_video_")
        try:
            clip_path  = os.path.join(tmp_dir, f"clip{video_ext}")
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir)

            with open(clip_path, "wb") as f:
                f.write(video_bytes)

            cmd = [
                "ffmpeg", "-i", clip_path,
                "-vf", f"fps={fps}",
                os.path.join(frames_dir, "frame_%04d.jpg"),
                "-hide_banner", "-loglevel", "error",
            ]
            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                raise RuntimeError("FFmpeg failed — clip may be corrupt or unsupported.")

            frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            if not frame_paths:
                raise RuntimeError("No frames extracted — clip may be too short.")

            features = []
            for path in frame_paths:
                try:
                    feat = self._encode_pil(Image.open(path).convert("RGB"))
                    features.append(feat)
                except Exception:
                    continue

            if not features:
                raise RuntimeError("Could not encode any frames.")

            results = self._search(np.vstack(features), top_k)
            return results, len(features)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)