import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import Sidebar from "./components/Sidebar.jsx";
import AskTab from "./components/AskTab.jsx";
import CompareTab from "./components/CompareTab.jsx";
import StatsTab from "./components/StatsTab.jsx";
import DocumentsTab from "./components/DocumentsTab.jsx";

const TABS = [
  { id: "ask", label: "💬 Hỏi đáp" },
  { id: "compare", label: "🔀 So sánh chiến lược" },
  { id: "stats", label: "📊 Thống kê" },
  { id: "docs", label: "📁 Tài liệu" },
];

function initialSettings(d) {
  return {
    strategy: d.strategy, mode: d.mode, topK: d.top_k, bm25TopN: d.bm25_top_n,
    faissTopN: d.faiss_top_n, rrfK: d.rrf_k, sources: [], useLlm: true,
  };
}

export default function App() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");
  const [settings, setSettings] = useState(null);
  // Tab hiện tại lưu trên URL (#ask, #docs...) để F5 không bị quay về tab đầu
  const [tab, setTabState] = useState(() => {
    const h = window.location.hash.slice(1);
    return TABS.some((t) => t.id === h) ? h : "ask";
  });
  const setTab = useCallback((id) => {
    setTabState(id);
    window.history.replaceState(null, "", `#${id}`);
  }, []);

  const loadInfo = useCallback(() => {
    setError("");
    return api.info()
      .then((data) => {
        setInfo(data);
        if (!data.strategies.length) setTab("docs");
        setSettings((prev) => {
          if (!prev) return initialSettings(data.defaults);
          // index vừa dựng lại: giữ lựa chọn cũ nếu còn hợp lệ
          const files = new Set(data.sources.map((s) => s.file));
          return {
            ...prev,
            strategy: data.strategies.includes(prev.strategy) ? prev.strategy
              : data.strategies.includes(data.defaults.strategy) ? data.defaults.strategy : data.strategies[0],
            sources: prev.sources.filter((f) => files.has(f)),
          };
        });
      })
      .catch((e) => { setError(e.message); throw e; });
  }, [setTab]);

  // Lần đầu mở trang: backend có thể còn đang nạp mô hình (~1 phút),
  // nên tự thử lại mỗi 3 giây (tối đa ~3 phút) thay vì báo lỗi ngay.
  const [waited, setWaited] = useState(0);
  const [gaveUp, setGaveUp] = useState(false);
  useEffect(() => {
    let stop = false;
    let timer;
    const started = Date.now();
    const attempt = () => loadInfo().catch(() => {
      if (stop) return;
      const elapsed = Date.now() - started;
      if (elapsed > 180000) { setGaveUp(true); return; }
      setWaited(Math.round(elapsed / 1000));
      timer = setTimeout(attempt, 3000);
    });
    attempt();
    return () => { stop = true; clearTimeout(timer); };
  }, [loadInfo]);

  if (error && (gaveUp || info)) {
    return (
      <div className="center">
        <div className="alert alert-error">
          <strong>Không kết nối được backend:</strong> {error}
          <p>Kiểm tra cửa sổ <b>Backend</b> có báo lỗi không, hoặc chạy lại:</p>
          <pre>cd backend{"\n"}uvicorn api.main:app --port 8000 --reload</pre>
          <button className="btn" onClick={() => loadInfo().catch(() => {})}>Thử lại</button>
        </div>
      </div>
    );
  }
  if (!info || !settings) {
    return (
      <div className="center">
        <div className="loading-card">
          <span className="spinner" aria-hidden="true" />
          <div>
            <b>Backend đang khởi động...</b>
            <div className="muted small">
              Lần đầu cần khoảng 1 phút để nạp mô hình. Trang sẽ tự mở khi sẵn sàng{waited ? ` (đã chờ ${waited}s)` : ""}.
            </div>
          </div>
        </div>
      </div>
    );
  }

  const hasIndex = info.strategies.length > 0;
  const needIndex = (
    <div className="empty-state">
      <div className="dropzone-icon">📂</div>
      <b>Chưa có tài liệu nào được index</b>
      <p className="muted">Vào tab Tài liệu để tải PDF lên, hệ thống sẽ tự dựng index.</p>
      <button className="btn primary" onClick={() => setTab("docs")}>Đến tab Tài liệu</button>
    </div>
  );

  return (
    <div className="layout">
      <Sidebar info={info} settings={settings} onChange={setSettings} />
      <main className="main">
        <header className="page-head">
          <h1>📚 Hybrid RAG</h1>
          <p className="muted">Hỏi đáp trên tài liệu PDF · BM25 + FAISS + RRF</p>
        </header>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.id} className={tab === t.id ? "tab active" : "tab"} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
        {/* Giữ tab Hỏi đáp / So sánh luôn mounted để không mất kết quả khi chuyển tab */}
        <div hidden={tab !== "ask"}>{hasIndex ? <AskTab settings={settings} info={info} /> : needIndex}</div>
        <div hidden={tab !== "compare"}>{hasIndex ? <CompareTab settings={settings} info={info} /> : needIndex}</div>
        {tab === "stats" && (hasIndex ? <StatsTab /> : needIndex)}
        {/* Luôn mounted để vẫn theo dõi tiến độ dựng index khi người dùng chuyển sang tab khác */}
        <div hidden={tab !== "docs"}><DocumentsTab onIndexChanged={() => loadInfo().catch(() => {})} /></div>
      </main>
    </div>
  );
}
