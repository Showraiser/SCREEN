---
title: SCREEN
emoji: 🎬
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 4.20.0
app_file: app.py
pinned: false
license: mit
---

# SCREEN — Cinema Recognition Engine

Identify any movie from a single screenshot using CLIP + FAISS visual search.

## Setup

Upload your index artefacts using Git LFS:

```bash
git lfs install
git lfs track "*.index" "*.joblib"
mkdir index
cp /path/to/screen_hnsw.index index/
cp /path/to/pca_model.joblib   index/
cp /path/to/metadata.csv       index/
git add . && git commit -m "Add index artefacts" && git push
```

The `inference.py` from the main repo is also needed — copy it into the `spaces/` folder.
