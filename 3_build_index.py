"""
Step 3 — Build a FAISS HNSW index from CLIP features.

Pipeline:
  1. Load all clip_features.npy files (and matching frame_names.npy).
  2. Reduce 512-D → 128-D with PCA.
  3. Build a FAISS HNSW flat index.
  4. Save index, PCA model, frame map, and a metadata CSV.
"""

import os
import csv
import argparse
import pickle

import numpy as np
import faiss
import joblib
from sklearn.decomposition import PCA


def load_movie_metadata(csv_path):
    metadata = {}
    if not csv_path or not os.path.exists(csv_path):
        return metadata
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["movie_name"]
            metadata[name] = {
                "year": row.get("year", ""),
                "director": row.get("director", ""),
                "genre": row.get("genre", ""),
            }
    return metadata


def collect_vectors(clip_dir, frame_root, movie_metadata):
    all_vectors = []
    frame_map = []     # (movie_name, frame_filename)
    metadata_rows = []
    faiss_id = 0

    for dirpath, _, filenames in os.walk(clip_dir):
        if "clip_features.npy" not in filenames:
            continue

        rel_path = os.path.relpath(dirpath, clip_dir)
        movie = os.path.basename(rel_path)

        vectors = np.load(os.path.join(dirpath, "clip_features.npy"))

        # Use saved frame names if available; fall back to positional index
        names_path = os.path.join(dirpath, "frame_names.npy")
        if os.path.exists(names_path):
            frame_names = list(np.load(names_path))
        else:
            frame_names = [f"{i + 1}.jpg" for i in range(vectors.shape[0])]

        all_vectors.append(vectors)
        meta = movie_metadata.get(movie, {"year": "", "director": "", "genre": ""})

        for i, fname in enumerate(frame_names):
            frame_path = os.path.join(frame_root, rel_path, fname)
            frame_map.append((movie, fname))
            metadata_rows.append([
                faiss_id, movie,
                meta["year"], meta["director"], meta["genre"],
                i, frame_path,
            ])
            faiss_id += 1

        print(f"  Loaded {vectors.shape[0]:>5} vectors  ← {rel_path}")

    return all_vectors, frame_map, metadata_rows


def build_index(all_vectors, target_dim, hnsw_m, ef_construction, ef_search):
    stacked = np.vstack(all_vectors).astype("float32")
    print(f"Total vectors: {stacked.shape[0]} × {stacked.shape[1]}D")

    print(f"Reducing to {target_dim}D with PCA...")
    pca = PCA(n_components=target_dim)
    reduced = pca.fit_transform(stacked).astype("float32")

    print("Building HNSW index...")
    index = faiss.IndexHNSWFlat(target_dim, hnsw_m)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(reduced)
    print(f"Index built — {index.ntotal} vectors")

    return index, pca


def main():
    parser = argparse.ArgumentParser(description="Build FAISS HNSW index from CLIP features.")
    parser.add_argument("--clip_dir", default="clip_features", help="Root folder of CLIP .npy files (default: ./clip_features).")
    parser.add_argument("--frame_root", default="frames", help="Root folder of extracted frames (default: ./frames).")
    parser.add_argument("--output_dir", default="index", help="Folder to save index and artefacts (default: ./index).")
    parser.add_argument("--movie_metadata", default="movie_metadata.csv", help="CSV with columns: movie_name, year, director, genre.")
    parser.add_argument("--target_dim", type=int, default=128, help="PCA target dimensions (default: 128).")
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    movie_metadata = load_movie_metadata(args.movie_metadata)
    print(f"Loaded metadata for {len(movie_metadata)} movies.")

    print("\nCollecting CLIP features...")
    all_vectors, frame_map, metadata_rows = collect_vectors(
        args.clip_dir, args.frame_root, movie_metadata
    )

    if not all_vectors:
        print("No clip_features.npy files found. Run step 2 first.")
        return

    index, pca = build_index(
        all_vectors, args.target_dim,
        args.hnsw_m, args.ef_construction, args.ef_search
    )

    # ── Save artefacts ──────────────────────────────────────────────────────
    index_path = os.path.join(args.output_dir, "screen_hnsw.index")
    map_path = os.path.join(args.output_dir, "screen_index_map.pkl")
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    pca_path = os.path.join(args.output_dir, "pca_model.joblib")

    faiss.write_index(index, index_path)
    with open(map_path, "wb") as f:
        pickle.dump(frame_map, f)
    joblib.dump(pca, pca_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["faiss_id", "movie_name", "year", "director", "genre", "frame_index", "frame_path"])
        writer.writerows(metadata_rows)

    print(f"\nSaved FAISS index  → {index_path}")
    print(f"Saved frame map    → {map_path}")
    print(f"Saved metadata CSV → {csv_path}")
    print(f"Saved PCA model    → {pca_path}")


if __name__ == "__main__":
    main()
