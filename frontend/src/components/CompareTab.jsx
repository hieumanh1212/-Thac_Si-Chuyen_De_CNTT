import { useMemo, useState } from "react";
import { api, searchParams } from "../api.js";
import { STRATEGY_INFO } from "./Sidebar.jsx";

// "YOLOv7: Trainable bag-of-..." -> "YOLOv7"; tên dài không có ":" thì cắt bớt
function shortName(src) {
  const ref = (src?.short_ref || src?.title || src?.file || "").trim();
  const head = ref.split(":")[0].trim();
  if (head && head.length <= 28) return head;
  return ref.length > 28 ? `${ref.slice(0, 26)}…` : ref;
}

// ". Implimentation Details" -> "Implimentation Details"
const cleanSection = (s) => (s || "").replace(/^[\s.·:-]+/, "").trim();
const firstPage = (p) => String(p ?? "").split("-")[0];
// Hai đoạn ở 2 chiến lược coi là "cùng chỗ" nếu cùng tài liệu + cùng trang bắt đầu
const placeKey = (c) => `${c.source}|${firstPage(c.pages)}`;

function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

// In đậm các từ trong câu hỏi xuất hiện trong đoạn văn
function Highlight({ text, terms }) {
  if (!terms.length) return text;
  const re = new RegExp(`(${terms.map(escapeRe).join("|")})`, "gi");
  return text.split(re).map((part, i) => (i % 2 === 1 ? <b key={i} className="cmp-hl">{part}</b> : part));
}

function CompareCard({ chunk, doc, terms, agree, total }) {
  const [open, setOpen] = useState(false);
  const section = cleanSection(chunk.section);
  const ranks = [
    chunk.bm25_rank != null && `Từ khóa #${chunk.bm25_rank}`,
    chunk.faiss_rank != null && `Ngữ nghĩa #${chunk.faiss_rank}`,
    `${chunk.text.length} ký tự`,
  ].filter(Boolean).join(" · ");
  return (
    <article className="cmp-card">
      <header className="cmp-head">
        <span className="cmp-rank">#{chunk.rank}</span>
        <span className="doc-chip" title={doc.full}>{doc.name}</span>
        <span className="cmp-page">tr.{chunk.pages}</span>
        {total > 1 && agree === total && (
          <span className="agree-badge" title="Các chiến lược khác cũng tìm ra trang này">✓ {agree}/{total}</span>
        )}
      </header>
      <div className="cmp-section" title={section}>{section || " "}</div>
      <p className={open ? "cmp-text open" : "cmp-text"}>
        <Highlight text={open ? chunk.text : chunk.text.replace(/\s+/g, " ")} terms={terms} />
      </p>
      <button type="button" className="link-btn" onClick={() => setOpen(!open)}>
        {open ? "Thu gọn" : "Xem toàn văn"}
      </button>
      <footer className="cmp-foot" title={`Điểm ${chunk.score.toFixed(4)}`}>{ranks}</footer>
    </article>
  );
}

export default function CompareTab({ settings, info }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    if (!question.trim()) {
      setError("Vui lòng nhập câu hỏi.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const { strategy: _ignored, ...params } = searchParams(settings);  // so sánh mọi chiến lược
      setData(await api.compare({ question: question.trim(), ...params }));
    } catch (err) {
      setError(err.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  // Màu + tên ngắn cho từng tài liệu (cố định theo thứ tự trong /api/info)
  const docs = useMemo(() => {
    const map = {};
    (info?.sources ?? []).forEach((src) => { map[src.file] = { name: shortName(src), full: src.title || src.file }; });
    return map;
  }, [info]);
  const docOf = (file) => docs[file] ?? { name: shortName({ file }), full: file };

  const view = useMemo(() => {
    if (!data) return null;
    const terms = [...new Set(data.question.split(/[\s,.;:?!()"']+/).filter((w) => w.length >= 3))];
    // Mỗi (tài liệu, trang) được bao nhiêu chiến lược tìm thấy
    const agreeCount = {};
    data.items.forEach((it) => {
      new Set(it.results.map(placeKey)).forEach((k) => { agreeCount[k] = (agreeCount[k] ?? 0) + 1; });
    });
    const shared = Object.values(agreeCount).filter((n) => n === data.items.length).length;
    return { terms, agreeCount, shared };
  }, [data]);

  return (
    <section>
      <p className="muted">Cùng một câu hỏi, so sánh kết quả truy xuất giữa các chiến lược chunking (không gọi LLM).</p>
      <form className="row" onSubmit={submit}>
        <input className="grow" value={question} placeholder="Câu hỏi so sánh"
          onChange={(e) => setQuestion(e.target.value)} />
        <button className="btn primary" disabled={loading}>
          {loading ? <><span className="spinner sm" aria-hidden="true" />Đang so sánh</> : "So sánh"}
        </button>
      </form>
      {error && <div className="alert alert-error">{error}</div>}

      {data && view && (
        <>
          <div className="cmp-overview muted small">
            {view.shared > 0
              ? <>Có <b>{view.shared}</b> trang được cả {data.items.length} chiến lược cùng tìm ra (đánh dấu ✓). </>
              : <>Các chiến lược không trùng trang nào. </>}
            Từ khóa trong câu hỏi được <b className="cmp-hl">in đậm</b>.
          </div>

          <div className="compare-grid aligned"
            style={{ "--cols": data.items.length, "--rows": 1 + Math.max(1, ...data.items.map((it) => it.results.length)) }}>
            {data.items.map((item) => {
              const meta = STRATEGY_INFO[item.strategy] ?? { label: item.strategy, desc: "" };
              const avgLen = item.results.length
                ? Math.round(item.results.reduce((sum, c) => sum + c.text.length, 0) / item.results.length) : 0;
              return (
                <div key={item.strategy} className="compare-col">
                  <div className="cmp-col-head">
                    <h3>{meta.label}</h3>
                    <p className="muted small">{meta.desc}</p>
                    <div className="cmp-stats">{Math.round(item.total_ms)} ms · trung bình {avgLen} ký tự/đoạn</div>
                  </div>
                  {item.results.length === 0 && <p className="muted">Không tìm thấy đoạn nào.</p>}
                  {item.results.map((c) => (
                    <CompareCard key={c.chunk_id} chunk={c} doc={docOf(c.source)}
                      terms={view.terms} agree={view.agreeCount[placeKey(c)]} total={data.items.length} />
                  ))}
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
