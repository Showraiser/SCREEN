import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import Head from "next/head";
import ResultCard from "../components/ResultCard";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const IMAGE_TYPES = { "image/*": [".jpg", ".jpeg", ".png", ".webp"] };
const VIDEO_TYPES = { "video/*": [".mp4", ".mkv", ".mov", ".avi", ".webm"] };

export default function Home() {
  const [mode,    setMode]    = useState("image");
  const [preview, setPreview] = useState(null);
  const [file,    setFile]    = useState(null);
  const [results, setResults] = useState([]);
  const [frames,  setFrames]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return;
    const f = accepted[0];
    setFile(f);
    setResults([]);
    setFrames(null);
    setError("");
    setPreview(
      mode === "image"
        ? { type: "image", url: URL.createObjectURL(f) }
        : { type: "video", url: URL.createObjectURL(f), name: f.name }
    );
  }, [mode]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: mode === "image" ? IMAGE_TYPES : VIDEO_TYPES,
    multiple: false,
  });

  const switchMode = (m) => {
    setMode(m);
    setFile(null);
    setPreview(null);
    setResults([]);
    setFrames(null);
    setError("");
  };

  const handleSearch = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResults([]);
    setFrames(null);

    try {
      const body     = new FormData();
      body.append("file", file);
      const endpoint = mode === "image" ? "/query/image" : "/query/video";
      const res      = await fetch(`${API_URL}${endpoint}?top_k=5`, { method: "POST", body });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      setResults(data.results);
      if (data.frames_used != null) setFrames(data.frames_used);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null); setPreview(null);
    setResults([]); setFrames(null); setError("");
  };

  return (
    <>
      <Head>
        <title>SCREEN — Cinema Recognition Engine</title>
        <meta name="description" content="Identify any movie from a screenshot or a clip." />
      </Head>

      <div className="page">

        {/* ── Header ── */}
        <header className="header">
          <div className="logo-row">
            <span className="logo-gem">◈</span>
            <span className="logo-word">SCREEN</span>
          </div>
          <h1 className="headline">Cinema Recognition Engine</h1>
          <p className="sub">Drop a frame or a clip. Find the film.</p>
        </header>

        {/* ── Mode tabs ── */}
        <div className="tabs">
          <button className={`tab ${mode === "image" ? "tab-on" : ""}`} onClick={() => switchMode("image")}>
            ▣ &nbsp;Image
          </button>
          <button className={`tab ${mode === "video" ? "tab-on" : ""}`} onClick={() => switchMode("video")}>
            ▶ &nbsp;Video clip
          </button>
        </div>

        {/* ── Upload ── */}
        <section>
          {!preview ? (
            <div {...getRootProps()} className={`zone ${isDragActive ? "zone-hot" : ""}`}>
              <input {...getInputProps()} />
              <div className="zone-icon">{mode === "image" ? "▧" : "▶"}</div>
              <p className="zone-main">
                {isDragActive ? "Release to load" : mode === "image" ? "Drop a screenshot here" : "Drop a video clip here"}
              </p>
              <p className="zone-hint">
                {mode === "image" ? "JPG · PNG · WEBP" : "MP4 · MKV · MOV · AVI · WEBM"}
              </p>
            </div>
          ) : (
            <div className="preview-wrap">
              {preview.type === "image"
                ? <img src={preview.url} alt="preview" className="preview-img" />
                : (
                  <div className="vid-wrap">
                    <video src={preview.url} controls className="preview-vid" />
                    <p className="vid-name">{preview.name}</p>
                  </div>
                )
              }
              <div className="action-row">
                <button className="btn-go" onClick={handleSearch} disabled={loading}>
                  {loading
                    ? <><span className="spin" /> {mode === "video" ? "Extracting frames…" : "Identifying…"}</>
                    : "Identify Film"
                  }
                </button>
                <button className="btn-clear" onClick={reset}>Clear</button>
              </div>
            </div>
          )}
        </section>

        {/* ── Error ── */}
        {error && <div className="err-box">⚠ &nbsp;{error}</div>}

        {/* ── Results ── */}
        {results.length > 0 && (
          <section className="results-wrap">
            <div className="results-head">
              <span className="results-label">Matches</span>
              <span className="results-count">
                {frames != null ? `${frames} frames analysed` : `${results.length} results`}
              </span>
            </div>
            <div className="results-list">
              {results.map((r, i) => <ResultCard key={i} result={r} rank={i} />)}
            </div>
          </section>
        )}

        <footer className="footer">SCREEN · CLIP + FAISS · Visual search</footer>
      </div>

      <style jsx>{`
        /* ── Page shell ── */
        .page {
          min-height: 100vh;
          max-width: 780px;
          margin: 0 auto;
          padding: 72px 28px 56px;
          display: flex;
          flex-direction: column;
          gap: 40px;
        }

        /* ── Header ── */
        .header { text-align: center; }
        .logo-row {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          margin-bottom: 18px;
        }
        .logo-gem {
          font-size: 26px;
          color: #e8d5a3;
        }
        .logo-word {
          font-family: 'DM Mono', monospace;
          font-size: 14px;
          letter-spacing: 0.4em;
          text-transform: uppercase;
          color: #f5f0e8;
          font-weight: 500;
        }
        .headline {
          font-family: 'DM Serif Display', serif;
          font-size: clamp(32px, 6vw, 52px);
          color: #f5f0e8;
          margin-bottom: 12px;
          line-height: 1.1;
          letter-spacing: -0.01em;
          font-weight: 400;
        }
        .sub {
          font-size: 15px;
          color: #888888;
          letter-spacing: 0.04em;
        }

        /* ── Tabs ── */
        .tabs {
          display: flex;
          gap: 6px;
          background: #141414;
          border: 1px solid #2e2e2e;
          border-radius: 8px;
          padding: 5px;
        }
        .tab {
          flex: 1;
          padding: 11px 16px;
          background: transparent;
          border: none;
          border-radius: 5px;
          cursor: pointer;
          font-family: 'DM Mono', monospace;
          font-size: 13px;
          letter-spacing: 0.06em;
          color: #888888;
          transition: background 0.15s, color 0.15s;
        }
        .tab-on {
          background: #e8d5a3;
          color: #080808;
          font-weight: 500;
        }
        .tab:not(.tab-on):hover { color: #f5f0e8; }

        /* ── Drop zone ── */
        .zone {
          border: 1px dashed #2e2e2e;
          border-radius: 8px;
          padding: 80px 32px;
          cursor: pointer;
          text-align: center;
          transition: border-color 0.2s, background 0.2s;
        }
        .zone:hover, .zone-hot {
          border-color: #e8d5a3;
          background: rgba(232, 213, 163, 0.04);
        }
        .zone-icon {
          font-size: 36px;
          color: #2e2e2e;
          margin-bottom: 20px;
          transition: color 0.2s;
        }
        .zone:hover .zone-icon, .zone-hot .zone-icon { color: #e8d5a3; }
        .zone-main {
          font-size: 17px;
          color: #f5f0e8;
          margin-bottom: 8px;
          font-family: 'DM Mono', monospace;
        }
        .zone-hint {
          font-size: 13px;
          color: #666666;
          letter-spacing: 0.06em;
        }

        /* ── Preview ── */
        .preview-wrap { display: flex; flex-direction: column; gap: 16px; }
        .preview-img {
          width: 100%;
          max-height: 420px;
          object-fit: cover;
          border-radius: 6px;
          border: 1px solid #2e2e2e;
        }
        .vid-wrap { display: flex; flex-direction: column; gap: 8px; }
        .preview-vid {
          width: 100%;
          max-height: 380px;
          border-radius: 6px;
          border: 1px solid #2e2e2e;
          background: #000;
        }
        .vid-name { font-size: 12px; color: #666666; }

        .action-row { display: flex; gap: 10px; }
        .btn-go {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 15px;
          background: #e8d5a3;
          color: #080808;
          border: none;
          border-radius: 5px;
          font-family: 'DM Mono', monospace;
          font-size: 14px;
          font-weight: 500;
          letter-spacing: 0.05em;
          cursor: pointer;
          transition: opacity 0.15s;
        }
        .btn-go:hover    { opacity: 0.88; }
        .btn-go:disabled { opacity: 0.45; cursor: default; }
        .btn-clear {
          padding: 15px 22px;
          background: transparent;
          color: #888888;
          border: 1px solid #2e2e2e;
          border-radius: 5px;
          font-family: 'DM Mono', monospace;
          font-size: 14px;
          cursor: pointer;
          transition: border-color 0.15s, color 0.15s;
        }
        .btn-clear:hover { border-color: #888888; color: #f5f0e8; }

        .spin {
          display: inline-block;
          width: 13px; height: 13px;
          border: 2px solid #080808;
          border-top-color: transparent;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* ── Error ── */
        .err-box {
          border: 1px solid #6b2828;
          background: #1a0808;
          color: #f08080;
          padding: 14px 18px;
          border-radius: 6px;
          font-size: 14px;
          font-family: 'DM Mono', monospace;
        }

        /* ── Results ── */
        .results-head {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          margin-bottom: 16px;
        }
        .results-label {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.18em;
          color: #666666;
          font-family: 'DM Mono', monospace;
        }
        .results-count {
          font-size: 12px;
          color: #888888;
          font-family: 'DM Mono', monospace;
        }
        .results-list { display: flex; flex-direction: column; gap: 10px; }

        /* ── Footer ── */
        .footer {
          text-align: center;
          font-size: 12px;
          color: #444444;
          letter-spacing: 0.08em;
          padding-top: 20px;
          border-top: 1px solid #1e1e1e;
          font-family: 'DM Mono', monospace;
        }
      `}</style>
    </>
  );
}