// Thanh cấu hình truy vấn: chia theo từng bước, mỗi lựa chọn có giải thích ngắn.

export const STRATEGY_INFO = {
  fixed: { label: "Fixed-size", desc: "Cắt đều theo số ký tự, có phần chồng lấn" },
  recursive: { label: "Recursive", desc: "Cắt theo đoạn → câu, giữ ranh giới mục" },
  semantic: { label: "Semantic", desc: "Cắt khi nội dung chuyển chủ đề (dùng embedding)" },
};

export const MODE_INFO = {
  hybrid: { label: "Kết hợp", full: "Hybrid (BM25 + FAISS → RRF)",
    desc: "Gộp kết quả từ khóa và ngữ nghĩa bằng RRF. Nên dùng mặc định." },
  bm25: { label: "Từ khóa", full: "Chỉ BM25 (từ khóa)",
    desc: "Khớp đúng từ ngữ. Tốt cho tên riêng, thuật ngữ, con số." },
  faiss: { label: "Ngữ nghĩa", full: "Chỉ FAISS (ngữ nghĩa)",
    desc: "So khớp theo ý nghĩa. Tốt khi câu hỏi diễn đạt khác tài liệu." },
};

export const MODE_LABELS = Object.fromEntries(Object.entries(MODE_INFO).map(([k, v]) => [k, v.full]));

function Step({ n, title, hint, children, aside }) {
  return (
    <section className="step">
      <div className="step-head">
        <span className="step-num">{n}</span>
        <span className="step-title">{title}</span>
        {aside}
      </div>
      {hint && <p className="step-hint">{hint}</p>}
      {children}
    </section>
  );
}

function SliderField({ label, hint, value, min, max, onChange, disabled }) {
  return (
    <div className={disabled ? "slider-field disabled" : "slider-field"}>
      <div className="slider-top">
        <span>{label}</span>
        <b>{value}</b>
      </div>
      <input type="range" min={min} max={max} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <p className="step-hint">{hint}</p>}
    </div>
  );
}

export default function Sidebar({ info, settings, onChange }) {
  const set = (key) => (value) => onChange((s) => ({ ...s, [key]: value }));
  const d = info.defaults;
  const hasIndex = info.strategies.length > 0;
  const allSources = settings.sources.length === 0;

  const toggleSource = (file) =>
    onChange((s) => ({
      ...s,
      sources: s.sources.includes(file) ? s.sources.filter((f) => f !== file) : [...s.sources, file],
    }));

  const resetDefaults = () =>
    onChange((s) => ({
      ...s, strategy: d.strategy, mode: d.mode, topK: d.top_k, bm25TopN: d.bm25_top_n,
      faissTopN: d.faiss_top_n, rrfK: d.rrf_k, sources: [],
    }));

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <h2>Cấu hình truy vấn</h2>
        <button className="link-btn" onClick={resetDefaults} title="Khôi phục tham số mặc định trong config.yaml">
          ↺ Mặc định
        </button>
      </div>

      {!hasIndex && (
        <div className="alert alert-warn small">Chưa có index. Hãy tải PDF lên ở tab <b>Tài liệu</b>.</div>
      )}

      <Step n={1} title="Chiến lược chunking" hint="Cách chia tài liệu thành các đoạn nhỏ để tìm kiếm.">
        <div className="option-cards">
          {info.strategies.map((s) => {
            const meta = STRATEGY_INFO[s] ?? { label: s, desc: "" };
            return (
              <button key={s} type="button" onClick={() => set("strategy")(s)}
                className={settings.strategy === s ? "option-card selected" : "option-card"}>
                <span className="option-top">
                  <span className="radio-dot" />
                  <b>{meta.label}</b>
                  {info.n_chunks?.[s] != null && <span className="badge">{info.n_chunks[s]} đoạn</span>}
                </span>
                <span className="option-desc">{meta.desc}</span>
              </button>
            );
          })}
        </div>
      </Step>

      <Step n={2} title="Cách tìm kiếm">
        <div className="segmented">
          {info.modes.map((m) => (
            <button key={m} type="button" onClick={() => set("mode")(m)}
              className={settings.mode === m ? "seg selected" : "seg"}>
              {MODE_INFO[m]?.label ?? m}
            </button>
          ))}
        </div>
        <p className="step-hint">{MODE_INFO[settings.mode]?.desc}</p>
      </Step>

      <Step n={3} title="Số đoạn gửi cho LLM (Top-K)">
        <div className="stepper">
          <button type="button" onClick={() => set("topK")(Math.max(1, settings.topK - 1))} aria-label="Giảm">−</button>
          <input type="range" min={1} max={10} value={settings.topK} onChange={(e) => set("topK")(Number(e.target.value))} />
          <button type="button" onClick={() => set("topK")(Math.min(10, settings.topK + 1))} aria-label="Tăng">+</button>
          <b className="stepper-value">{settings.topK}</b>
        </div>
        <p className="step-hint">Nhiều đoạn: đủ ý hơn nhưng tốn token và dễ lẫn thông tin nhiễu. Khuyên dùng 3–5.</p>
      </Step>

      <Step n={4} title="Phạm vi tài liệu">
        <div className="chips">
          <button type="button" className={allSources ? "chip selected" : "chip"}
            onClick={() => onChange((s) => ({ ...s, sources: [] }))}>
            Tất cả ({info.sources.length})
          </button>
          {info.sources.map((s) => (
            <button key={s.file} type="button" title={`${s.title}\n${s.file}`}
              className={settings.sources.includes(s.file) ? "chip selected" : "chip"}
              onClick={() => toggleSource(s.file)}>
              {s.short_ref}
            </button>
          ))}
        </div>
      </Step>

      <Step n={5} title="Sinh câu trả lời bằng LLM"
        aside={
          <label className="switch" title="Tắt để chỉ xem các đoạn tìm được">
            <input type="checkbox" checked={settings.useLlm} onChange={(e) => set("useLlm")(e.target.checked)} />
            <span className="slider" />
          </label>
        }>
        <p className={info.llm.ready ? "llm-status ok" : "llm-status warn"}>
          <span className="dot" />
          {info.llm.ready ? "Đã có API key" : "Chưa có API key - chỉ hiển thị đoạn tìm được"}
        </p>
        <p className="step-hint"><code>{info.llm.provider}</code> / <code>{info.llm.model}</code></p>
      </Step>

      <details className="advanced">
        <summary>Tham số nâng cao <span className="muted small">(cho thí nghiệm)</span></summary>
        <SliderField label="BM25 Top-N" value={settings.bm25TopN} min={1} max={30} onChange={set("bm25TopN")}
          disabled={settings.mode === "faiss"} hint="Số ứng viên lấy từ BM25 trước khi hợp nhất." />
        <SliderField label="FAISS Top-N" value={settings.faissTopN} min={1} max={30} onChange={set("faissTopN")}
          disabled={settings.mode === "bm25"} hint="Số ứng viên lấy từ FAISS trước khi hợp nhất." />
        <SliderField label="RRF k" value={settings.rrfK} min={1} max={200} onChange={set("rrfK")}
          disabled={settings.mode !== "hybrid"}
          hint="Hằng số làm mượt của RRF: điểm = Σ 1/(k + hạng). Mặc định 60." />
      </details>
    </aside>
  );
}
