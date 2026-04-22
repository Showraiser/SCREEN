"""
SCREEN — Hugging Face Spaces (Gradio) demo.

Upload a screenshot OR a video clip and SCREEN identifies the film.

Place index artefacts in ./index/:
    index/screen_hnsw.index
    index/pca_model.joblib
    index/metadata.csv
"""

import io
import os
import sys

import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from inference import SCREENInference  # noqa: E402

INDEX_DIR = os.getenv("INDEX_DIR", "index")

print("Loading SCREEN…")
try:
    screen = SCREENInference(index_dir=INDEX_DIR)
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    screen = None


# ── helpers ───────────────────────────────────────────────────────────────────
def _format_results(results: list[dict], frames_used: int | None = None) -> str:
    if not results:
        return "No matches found."

    lines = []
    for i, r in enumerate(results, 1):
        conf = int(r["confidence"] * 100)
        bar  = "█" * (conf // 5) + "░" * (20 - conf // 5)
        lines.append(
            f"{'🥇' if i == 1 else f'#{i}'}  **{r['movie_name']}** ({r['year']})  \n"
            f"   Director: {r['director']}  ·  Genre: {r['genre']}  \n"
            f"   `{bar}` {conf}%"
        )

    footer = f"\n\n*{frames_used} frames analysed*" if frames_used else ""
    return "\n\n".join(lines) + footer


# ── query functions ───────────────────────────────────────────────────────────
def identify_image(image: Image.Image):
    if screen is None:
        return "⚠️ Index not loaded. See README for setup."
    if image is None:
        return "Please upload a screenshot."

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    results = screen.query_image(buf.getvalue(), top_k=5)
    return _format_results(results)


def identify_video(video_path: str):
    if screen is None:
        return "⚠️ Index not loaded. See README for setup."
    if not video_path:
        return "Please upload a video clip."

    ext = os.path.splitext(video_path)[1].lower() or ".mp4"
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    try:
        results, n_frames = screen.query_video(
            video_bytes, video_ext=ext, fps=1, top_k=5
        )
    except RuntimeError as e:
        return f"⚠️ {e}"

    return _format_results(results, frames_used=n_frames)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="SCREEN — Cinema Recognition Engine",
    theme=gr.themes.Base(
        primary_hue="stone",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("DM Mono"), "monospace"],
    ),
    css="""
        #title  { text-align: center; font-size: 2rem; margin-bottom: 4px; }
        #sub    { text-align: center; color: #888; margin-bottom: 24px; }
        .footer { text-align: center; font-size: 11px; color: #555; margin-top: 24px; }
    """,
) as demo:
    gr.Markdown("# ◈ SCREEN", elem_id="title")
    gr.Markdown(
        "**Cinema Recognition Engine** — upload a screenshot or a short clip to identify the movie.",
        elem_id="sub",
    )

    with gr.Tabs():
        # ── Image tab ──────────────────────────────────────────────────────
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Movie screenshot", height=320)
                    img_btn   = gr.Button("Identify Film", variant="primary")
                with gr.Column():
                    img_out = gr.Markdown(value="*Upload a screenshot to begin.*")

            img_btn.click(fn=identify_image, inputs=img_input, outputs=img_out)
            img_input.change(fn=identify_image, inputs=img_input, outputs=img_out)

        # ── Video tab ──────────────────────────────────────────────────────
        with gr.Tab("Video clip"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Movie clip (MP4 / MKV / MOV / AVI)")
                    vid_btn   = gr.Button("Identify Film", variant="primary")
                with gr.Column():
                    vid_out = gr.Markdown(value="*Upload a clip to begin.*")

            vid_btn.click(fn=identify_video, inputs=vid_input, outputs=vid_out)

    gr.Markdown(
        "Built with [CLIP](https://github.com/openai/CLIP) + "
        "[FAISS](https://github.com/facebookresearch/faiss) · "
        "[GitHub](https://github.com/YOUR_USERNAME/SCREEN)",
        elem_classes=["footer"],
    )

if __name__ == "__main__":
    demo.launch()