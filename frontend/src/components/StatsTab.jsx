import { useEffect, useState } from "react";
import { api } from "../api.js";

const COLORS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"];

function fmt(v, digits = 0) {
  return v == null ? "-" : typeof v === "number" ? v.toFixed(digits) : String(v);
}

// Biểu đồ cột nhóm: mỗi khoảng độ dài có 1 cột cho mỗi chiến lược
function Histogram({ strategies }) {
  const bins = Object.keys(strategies[0]?.length_histogram ?? {});
  const max = Math.max(1, ...strategies.flatMap((s) => Object.values(s.length_histogram)));
  return (
    <div>
      <div className="legend">
        {strategies.map((s, i) => (
          <span key={s.strategy}><i style={{ background: COLORS[i % COLORS.length] }} />{s.strategy}</span>
        ))}
      </div>
      <div className="hist">
        {bins.map((bin) => (
          <div key={bin} className="hist-group">
            <div className="hist-bars">
              {strategies.map((s, i) => {
                const n = s.length_histogram[bin];
                return (
                  <div key={s.strategy} className="hist-bar" title={`${s.strategy}: ${n} chunk`}
                    style={{ height: `${(n / max) * 100}%`, background: COLORS[i % COLORS.length] }}>
                    {n > 0 && <span>{n}</span>}
                  </div>
                );
              })}
            </div>
            <div className="hist-label">{bin}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function StatsTab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.stats().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-error">{error}</div>;
  if (!data) return <p className="muted">Đang tải thống kê...</p>;

  return (
    <section>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Chiến lược</th><th>Số chunk</th><th>Độ dài TB</th><th>Trung vị</th><th>Min</th><th>Max</th>
              <th>Chunk vắt 2 trang</th><th>Build (s)</th><th>Tham số</th>
            </tr>
          </thead>
          <tbody>
            {data.strategies.map((s) => (
              <tr key={s.strategy}>
                <td><b>{s.strategy}</b></td><td>{fmt(s.n_chunks)}</td><td>{fmt(s.avg_chars, 1)}</td>
                <td>{fmt(s.median_chars)}</td><td>{fmt(s.min_chars)}</td><td>{fmt(s.max_chars)}</td>
                <td>{fmt(s.multi_page_chunks)}</td><td>{fmt(s.build_seconds, 2)}</td>
                <td className="small"><code>{JSON.stringify(s.params)}</code></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>Phân bố độ dài chunk (ký tự)</h3>
      <Histogram strategies={data.strategies} />

      <p className="muted small">
        Embedding: <code>{data.embedding_model}</code> (dim={data.embedding_dim}) · Tài liệu: {data.documents.join(", ")}
      </p>
    </section>
  );
}
