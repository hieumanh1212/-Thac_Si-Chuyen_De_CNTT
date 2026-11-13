import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { api, searchParams } from "../api.js";
import ChunkCard from "./ChunkCard.jsx";
import { MODE_INFO, MODE_LABELS, STRATEGY_INFO } from "./Sidebar.jsx";

// Dòng tóm tắt cấu hình đang dùng, để người dùng biết câu hỏi sẽ được xử lý thế nào
function SettingsSummary({ settings, info }) {
  const scope = settings.sources.length
    ? info.sources.filter((s) => settings.sources.includes(s.file)).map((s) => s.short_ref).join(", ")
    : `Tất cả tài liệu (${info.sources.length})`;
  return (
    <div className="summary">
      <span className="muted small">Đang dùng:</span>
      <span className="pill">{STRATEGY_INFO[settings.strategy]?.label ?? settings.strategy}</span>
      <span className="pill">{MODE_INFO[settings.mode]?.label ?? settings.mode}</span>
      <span className="pill">Top-{settings.topK}</span>
      <span className="pill">{scope}</span>
      <span className={settings.useLlm && info.llm.ready ? "pill ok" : "pill"}>
        {settings.useLlm ? (info.llm.ready ? "Có LLM" : "LLM chưa có key") : "Không gọi LLM"}
      </span>
    </div>
  );
}

const META_LABELS = {
  title: "Tiêu đề", authors: "Tác giả", year: "Năm", venue: "Nơi công bố", document_type: "Loại",
  doi_url: "DOI/URL", source_url: "URL", course: "Học phần",
};

function friendlyLlmError(msg) {
  if (/api_key|auth|credential|401|403/i.test(msg)) {
    return "Chưa cấu hình API key cho LLM (file backend/.env) nên chưa sinh được câu trả lời. "
      + "Bên dưới vẫn là kết quả truy xuất.";
  }
  return `Gọi LLM thất bại: ${msg}. Bên dưới vẫn là kết quả truy xuất.`;
}

function SourcesTable({ sources }) {
  const cols = Object.keys(META_LABELS).filter((k) => sources.some((s) => s.metadata[k]));
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr><th>File</th>{cols.map((c) => <th key={c}>{META_LABELS[c]}</th>)}<th>Trang / mục</th></tr>
        </thead>
        <tbody>
          {sources.map((s) => (
            <tr key={s.file}>
              <td>{s.file}</td>
              {cols.map((c) => {
                const v = s.metadata[c];
                const text = Array.isArray(v) ? v.join(", ") : String(v ?? "");
                return <td key={c}>{/^https?:\/\//.test(text) ? <a href={text} target="_blank" rel="noreferrer">{text}</a> : text}</td>;
              })}
              <td>{s.locations.join("; ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AskTab({ settings, info }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const submit = async () => {
    if (!question.trim()) {
      setError("Vui lòng nhập câu hỏi.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      setResult(await api.ask({ question: question.trim(), generate: settings.useLlm, ...searchParams(settings) }));
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const t = result?.timings_ms ?? {};
  return (
    <section>
      <SettingsSummary settings={settings} info={info} />
      <label className="field">
        <span>Câu hỏi</span>
        <textarea rows={3} value={question} placeholder="Ví dụ: Phương pháp A và B khác nhau như thế nào?"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(); }} />
      </label>
      <button className="btn primary" onClick={submit} disabled={loading}>
        {loading ? "Đang tìm kiếm và sinh câu trả lời..." : "Hỏi"}
      </button>
      <span className="muted small hint">Ctrl + Enter để gửi</span>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="result">
          {result.llm_error && <div className="alert alert-warn">{friendlyLlmError(result.llm_error)}</div>}

          {result.answer && (
            <>
              <h3>Câu trả lời</h3>
              <div className="answer"><ReactMarkdown>{result.answer_rendered}</ReactMarkdown></div>
              <h3>Tài liệu đã sử dụng</h3>
              <SourcesTable sources={result.sources_used} />
            </>
          )}

          <div className="metrics">
            <div className="metric"><span>Thời gian tìm kiếm</span><b>{Math.round(t.retrieval_total ?? 0)} ms</b></div>
            <div className="metric"><span>Thời gian LLM</span><b>{t.llm ? `${(t.llm / 1000).toFixed(1)} s` : "-"}</b></div>
            <div className="metric"><span>Token context (ước lượng)</span><b>{result.context_tokens}</b></div>
          </div>
          <p className="muted small">
            Chiến lược chunking: <b>{result.strategy}</b> · Chế độ tìm kiếm: <b>{MODE_LABELS[result.mode]}</b>
          </p>

          <h3>Top-{result.contexts.length} đoạn truy xuất</h3>
          <p className="muted small">✅ = được LLM trích dẫn trong câu trả lời</p>
          {result.contexts.map((c, i) => <ChunkCard key={c.chunk_id} chunk={c} index={i + 1} />)}
        </div>
      )}
    </section>
  );
}
