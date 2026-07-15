# SCREEN 🎬

**Cinema Recognition Engine** — identify any movie from a single screenshot using CLIP embeddings + FAISS visual search. No audio, no subtitles, purely visual.

---

## Architecture

```
Offline pipeline
  1_extract_frames.py  →  frames/          FFmpeg, 1 fps
  2_clip_extraction.py →  clip_features/   CLIP ViT-B/32, 512-D
  3_build_index.py     →  index/           PCA 512→128-D + FAISS HNSW
                              ↓
                    FastAPI backend   (backend/)
                    POST /query  ←  image upload
                       ↙             ↘
             Next.js frontend     HF Spaces (Gradio)
              (frontend/)             (spaces/)
```

---

## Quick start

### Requirements
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) on `PATH`
- CUDA GPU optional (speeds up CLIP encoding)

### 1. Build the index (offline, one-time)

```bash
pip install -r requirements.txt

python 1_extract_frames.py  --video_dir /path/to/movies  --output_dir frames
python 2_clip_extraction.py --frames_dir frames           --output_dir clip_features
python 3_build_index.py     --clip_dir clip_features      --frame_root frames \
                             --movie_metadata movie_metadata.csv --output_dir index
```

Fill in `movie_metadata.csv` with your library first (see sample file).

### 2. Run the backend

```bash
cd backend
pip install -r requirements.txt
INDEX_DIR=../index uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API explorer.

### 3. Run the Next.js frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`, drag a screenshot, hit **Identify Film**.

### 4. Or run the Gradio demo (no frontend needed)

```bash
cd spaces
pip install -r requirements.txt
cp ../backend/inference.py .        # spaces reuses the inference module
INDEX_DIR=../index python app.py
```

---

## Deploying on Hugging Face Spaces

```bash
# 1. Create a new Space at huggingface.co/new-space
#    SDK: Gradio, visibility: Public

# 2. Clone it and copy files
git clone https://huggingface.co/spaces/YOUR_USERNAME/SCREEN
cp spaces/* SCREEN/
cp backend/inference.py SCREEN/

# 3. Upload index artefacts with Git LFS
cd SCREEN
git lfs install
git lfs track "*.index" "*.joblib"
mkdir index
cp /path/to/index/* index/

git add . && git commit -m "Initial deploy" && git push
```

The Space builds automatically. Share the URL — anyone can drop in a screenshot.

---

## Deploying the FastAPI backend

**Docker:**
```bash
cd backend
cp -r ../index .          # copy index artefacts in
docker build -t screen-api .
docker run -p 8000:8000 screen-api
```

**Render / Railway / Fly.io:**
Point to `backend/` as the root, set env var `INDEX_DIR=/app/index`, start command:
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Project structure

```
SCREEN/
├── 1_extract_frames.py       ← Step 1: extract frames
├── 2_clip_extraction.py      ← Step 2: CLIP encoding
├── 3_build_index.py          ← Step 3: PCA + FAISS index
├── 4_query.py                ← Step 4: CLI query tool
├── movie_metadata.csv        ← Your library metadata (edit this)
├── requirements.txt          ← Offline pipeline deps
│
├── backend/
│   ├── main.py               ← FastAPI app
│   ├── inference.py          ← CLIP + FAISS inference engine
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── pages/
│   │   ├── _app.js
│   │   └── index.js          ← Main upload UI
│   ├── components/
│   │   └── ResultCard.jsx    ← Per-result card with confidence bar
│   ├── styles/globals.css    ← Film grain, DM Serif, dark theme
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
│
└── spaces/
    ├── app.py                ← Gradio demo (reuses inference.py)
    ├── requirements.txt
    └── README.md             ← HF Spaces YAML metadata header
```

> Generated folders (`frames/`, `clip_features/`, `index/`) are git-ignored — they can reach tens of GB.

---

## API reference

### `POST /query?top_k=5`

| Field | Value |
|---|---|
| Content-Type | `multipart/form-data` |
| Body | `file`: any `image/*` (JPEG, PNG, WEBP) |

**Response:**
```json
{
  "results": [
    {
      "movie_name": "Inception",
      "year": "2010",
      "director": "Christopher Nolan",
      "genre": "Sci-Fi",
      "confidence": 0.847
    }
  ]
}
```

### `GET /health`
```json
{ "status": "ok", "vectors_indexed": 182400, "device": "cuda" }
```

---

## movie_metadata.csv format

| Column | Description |
|---|---|
| `movie_name` | Must match the video filename (without extension) |
| `year` | Release year |
| `director` | Director name |
| `genre` | Genre |

```
movie_name,year,director,genre
Inception,2010,Christopher Nolan,Sci-Fi
The Dark Knight,2008,Christopher Nolan,Action
```

---

## Notes

- Confidence is computed as `1 / (1 + L2_distance)` — higher is better, max 1.0.
- HNSW `ef_search` (default 50) trades recall for speed — increase for better accuracy on large libraries.
- PCA target dimension (default 128) can be tuned via `--target_dim` in step 3.
- The larger and more varied your library, the more frames per movie help accuracy.
- For the HF Spaces deploy, `.index` and `.joblib` files must be tracked with Git LFS or they will be corrupted on push.
