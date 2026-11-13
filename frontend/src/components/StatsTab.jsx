import { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import { STRATEGY_INFO } from "./Sidebar.jsx";

const label = (s) => STRATEGY_INFO[s]?.label ?? s;
const TAGLINE = { fixed: "Cắt đều từng khúc", recursive: "Cắt theo mục, đoạn văn", semantic: "Cắt theo ý" };
const fmt = (v, digits = 0) => (v == null ? "-" : Number(v).toLocaleString("vi-VN", { maximumFractionDigits: digits }));
const pct = (a, b) => (b ? Math.round((a / b) * 100) : 0);

function fmtDuration(sec) {
  if (sec == null) return "-";
  const s = Math.round(sec);
  return s < 60 ? `${s} giây` : `${Math.floor(s / 60)} phút ${s % 60} giây`;
}

// Nhãn điểm mạnh: chiến lược nào tốt nhất ở chỉ số nào
function strengths(list) {
  const tags = Object.fromEntries(list.map((s) => [s.strategy, []]));
  const best = (f, text) => {
    const v = Math.min(...list.map(f));
    list.filter((s) => f(s) === v).forEach((s) => tags[s.strategy].push(text));
  };
  best((s) => s.build_seconds ?? Infinity, "Dựng nhanh nhất");
  best((s) => pct(s.multi_page_chunks, s.n_chunks), "Ít vắt trang nhất");
  best((s) => s.length_histogram["0-200"] ?? 0, "Ít đoạn vụn nhất");
  return tags;
}

export default function StatsTab() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.stats().then(setData).catch((e) => setError(e.message));
  }, []);

  const list = useMemo(() => (data?.strategies ?? []).filter((s) => s.n_chunks), [data]);
  const tags = useMemo(() => strengths(list), [list]);

  if (error) return <div className="alert alert-error">{error}</div>;
  if (!data) return <p className="muted">Đang tải thống kê...</p>;

  return (
    <section className="stats">
      <div className="st-grid" style={{ "--cols": list.length }}>
        {list.map((s) => {
          return (
            <article key={s.strategy} className="st-card">
              <h3>{label(s.strategy)}</h3>
              <div className="muted small">{TAGLINE[s.strategy] ?? ""}</div>

              <div className="st-nums">
                <div><b>{fmt(s.n_chunks)}</b><span>đoạn</span></div>
                <div><b>{fmt(s.avg_chars)}</b><span>ký tự / đoạn</span></div>
                <div title={fmtDuration(s.build_seconds)}><b>{s.build_seconds >= 60 ? `${Math.floor(s.build_seconds / 60)}:${String(Math.round(s.build_seconds % 60)).padStart(2, "0")}` : fmt(s.build_seconds)}</b>
                  <span>{s.build_seconds >= 60 ? "phút dựng index" : "giây dựng index"}</span></div>
              </div>

              <div className="st-tags">
                {(tags[s.strategy] ?? []).map((t) => <span key={t} className="st-tag">✓ {t}</span>)}
              </div>
            </article>
          );
        })}
      </div>

      <div className="stat-raw">
        <h3>Số liệu chi tiết</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Chiến lược</th><th>Số đoạn</th><th>Độ dài TB</th><th>Ngắn nhất</th><th>Dài nhất</th>
                <th>Đoạn &lt; 200 ký tự</th><th>Vắt 2 trang</th><th>Dựng index</th>
              </tr>
            </thead>
            <tbody>
              {list.map((s) => (
                <tr key={s.strategy}>
                  <td><b>{label(s.strategy)}</b></td><td>{s.n_chunks}</td><td>{fmt(s.avg_chars)}</td>
                  <td>{s.min_chars}</td><td>{s.max_chars}</td>
                  <td>{s.length_histogram["0-200"] ?? 0} ({pct(s.length_histogram["0-200"] ?? 0, s.n_chunks)}%)</td>
                  <td>{s.multi_page_chunks} ({pct(s.multi_page_chunks, s.n_chunks)}%)</td>
                  <td>{fmt(s.build_seconds, 1)} s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
