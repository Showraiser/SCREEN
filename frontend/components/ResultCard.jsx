export default function ResultCard({ result, rank }) {
  const raw = result.confidence * 100;
  const pct = Math.min(Math.round(raw), 100);
  const isTop = rank === 0;

  return (
    <div className="card" style={{ animationDelay: `${rank * 90}ms` }}>

      {/* left: rank */}
      <div className="rank-col">
        <span className="rank-num">{isTop ? "★" : `#${rank + 1}`}</span>
      </div>

      {/* centre: info */}
      <div className="info-col">
        <div className="title">{result.movie_name}</div>
        <div className="meta">
          {result.year     && <span className="pill pill-dim">{result.year}</span>}
          {result.genre    && <span className="pill pill-dim">{result.genre}</span>}
          {result.director && <span className="pill pill-gold">dir. {result.director}</span>}
        </div>
      </div>

      {/* right: confidence */}
      <div className="conf-col">
        <div className="conf-pct">{Math.round(raw)}%</div>
        <div className="bar-track">
          <div className="bar-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="conf-label">match</div>
      </div>

      <style jsx>{`
        .card {
          display: flex;
          align-items: center;
          gap: 20px;
          padding: 22px 24px;
          background: #1a1a1a;
          border: 1px solid ${isTop ? "#c9a96e" : "#2e2e2e"};
          border-left: 3px solid ${isTop ? "#e8d5a3" : "#2e2e2e"};
          border-radius: 6px;
          animation: fadeUp 0.45s ease forwards;
          opacity: 0;
          transition: border-color 0.2s, background 0.2s;
        }
        .card:hover {
          background: #202020;
          border-color: #c9a96e;
          border-left-color: #e8d5a3;
        }

        .rank-col {
          min-width: 32px;
          text-align: center;
        }
        .rank-num {
          font-family: 'DM Mono', monospace;
          font-size: ${isTop ? "20px" : "13px"};
          color: ${isTop ? "#e8d5a3" : "#666666"};
          font-weight: 500;
        }

        .info-col {
          flex: 1;
          min-width: 0;
        }
        .title {
          font-family: 'DM Serif Display', serif;
          font-size: 22px;
          color: #f5f0e8;
          margin-bottom: 10px;
          line-height: 1.25;
          letter-spacing: 0.01em;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .meta {
          display: flex;
          flex-wrap: wrap;
          gap: 7px;
        }
        .pill {
          font-family: 'DM Mono', monospace;
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.09em;
          padding: 3px 9px;
          border-radius: 3px;
          border: 1px solid transparent;
        }
        .pill-dim {
          color: #aaaaaa;
          border-color: #333333;
          background: #111111;
        }
        .pill-gold {
          color: #e8d5a3;
          border-color: #5a4a2a;
          background: #1e1810;
        }

        .conf-col {
          text-align: right;
          min-width: 70px;
        }
        .conf-pct {
          font-family: 'DM Mono', monospace;
          font-size: 22px;
          font-weight: 500;
          color: #e8d5a3;
          line-height: 1;
          margin-bottom: 6px;
        }
        .bar-track {
          height: 3px;
          background: #2e2e2e;
          border-radius: 2px;
          overflow: hidden;
          width: 70px;
          margin-bottom: 5px;
        }
        .bar-fill {
          height: 100%;
          background: linear-gradient(90deg, #c9a96e, #e8d5a3);
          border-radius: 2px;
          transition: width 1s cubic-bezier(0.23, 1, 0.32, 1);
        }
        .conf-label {
          font-family: 'DM Mono', monospace;
          font-size: 10px;
          color: #666666;
          text-transform: uppercase;
          letter-spacing: 0.1em;
        }
      `}</style>
    </div>
  );
}
