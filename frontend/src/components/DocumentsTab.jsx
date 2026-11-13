import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useConfirm } from "./ConfirmDialog.jsx";

// Các trường thông tin trích dẫn hiển thị trên form (phần còn lại vẫn sửa được trong sources.yaml)
const META_FORM = [
  { key: "short_ref", label: "Tên ngắn trong trích dẫn", placeholder: "Vaswani et al., 2017", wide: true },
  { key: "title", label: "Tiêu đề", placeholder: "Attention Is All You Need", wide: true },
  { key: "authors", label: "Tác giả", placeholder: "Vaswani et al." },
  { key: "year", label: "Năm", placeholder: "2017" },
  { key: "venue", label: "Nơi công bố", placeholder: "NeurIPS" },
  { key: "doi_url", label: "DOI / URL", placeholder: "https://arxiv.org/abs/..." },
];

// "2026-11-12T14:33:12" -> "14:33:12 12/11/2026"
const fmtDateTime = (iso) => {
  const m = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}:\d{2}:\d{2})/.exec(iso || "");
  return m ? `${m[4]} ${m[3]}/${m[2]}/${m[1]}` : iso;
};

const fmtSize = (kb) => (kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${Math.round(kb)} KB`);

function MetaForm({ value, onChange }) {
  return (
    <div className="meta-form">
      {META_FORM.map((f) => (
        <label key={f.key} className={f.wide ? "field wide" : "field"}>
          <span>{f.label}</span>
          <input value={value[f.key] ?? ""} placeholder={f.placeholder}
            onChange={(e) => onChange({ ...value, [f.key]: e.target.value })} />
        </label>
      ))}
    </div>
  );
}

function JobStatus({ job, needsRebuild, onRebuild, busy }) {
  if (job?.status === "running") {
    return (
      <div className="loading-card">
        <div className="spinner" aria-hidden="true" />
        <div>
          <b>Đang dựng index...</b>
          <p className="muted small">{job.message} Trong lúc chờ vẫn hỏi đáp được bằng index cũ.</p>
        </div>
      </div>
    );
  }
  return (
    <>
      {job?.status === "error" && (
        <div className="alert alert-error"><b>Dựng index thất bại.</b> {job.error}</div>
      )}
      {job?.status === "done" && !needsRebuild && (
        <div className="alert alert-ok">✓ {job.message}</div>
      )}
      {needsRebuild && (
        <div className="alert alert-warn job-row">
          <span>Danh sách hoặc thông tin tài liệu đã thay đổi so với index hiện tại.</span>
          <button className="btn primary" onClick={onRebuild} disabled={busy}>Dựng lại index</button>
        </div>
      )}
    </>
  );
}

export default function DocumentsTab({ onIndexChanged }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [queue, setQueue] = useState([]);           // file chờ upload: {id, file, meta, status, message}
  const [overwrite, setOverwrite] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [editing, setEditing] = useState(null);      // {file, meta}
  const [deleting, setDeleting] = useState(null);    // tên file đang xóa
  const [confirm, confirmDialog] = useConfirm();
  const inputRef = useRef(null);
  const wasRunning = useRef(false);

  const refresh = useCallback(() => {
    return api.documents().then((d) => { setData(d); setError(""); return d; }).catch((e) => setError(e.message));
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  // Theo dõi tiến độ khi đang dựng index; xong thì tải lại danh sách + báo App nạp lại cấu hình
  const running = data?.job?.status === "running";
  useEffect(() => {
    if (running) {
      wasRunning.current = true;
      const t = setInterval(() => {
        api.indexStatus().then((job) => {
          setData((d) => ({ ...d, job }));
          if (job.status !== "running") refresh();
        }).catch(() => {});
      }, 1500);
      return () => clearInterval(t);
    }
    if (wasRunning.current) {
      wasRunning.current = false;
      onIndexChanged?.();
    }
  }, [running, refresh, onIndexChanged]);

  const busy = running || uploading;

  const addFiles = (files) => {
    const pdfs = [...files].filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length < files.length) setError("Chỉ nhận file PDF - đã bỏ qua các file khác.");
    const maxMb = data?.max_upload_mb ?? 50;
    setQueue((q) => [
      ...q,
      ...pdfs.map((file) => ({
        id: `${file.name}-${file.size}-${Math.random()}`, file, meta: {}, open: false,
        status: file.size > maxMb * 1024 * 1024 ? "error" : "pending",
        message: file.size > maxMb * 1024 * 1024 ? `Lớn hơn ${maxMb} MB` : "",
      })),
    ]);
  };

  const updateItem = (id, patch) => setQueue((q) => q.map((it) => (it.id === id ? { ...it, ...patch } : it)));

  const uploadAll = async () => {
    const pending = queue.filter((it) => it.status === "pending" || it.status === "error");
    if (!pending.length) return;
    setUploading(true);
    let ok = 0;
    for (const it of pending) {
      updateItem(it.id, { status: "uploading", message: "" });
      try {
        const res = await api.upload(it.file, it.meta, { overwrite, rebuild: false });
        ok += 1;
        updateItem(it.id, { status: "done", message: res.warning || `${res.pages} trang` });
      } catch (e) {
        updateItem(it.id, { status: "error", message: e.message });
      }
    }
    if (ok) {
      try { await api.rebuild(); } catch { /* đã có job đang chạy */ }
      setQueue((q) => q.filter((it) => it.status !== "done" || it.message.includes("scan")));
    }
    await refresh();
    setUploading(false);
  };

  const remove = async (doc) => {
    const name = doc.metadata.short_ref || doc.metadata.title || doc.file;
    const ok = await confirm({
      title: "Xóa tài liệu?",
      message: (
        <>
          <p>Bạn sắp xóa <b>{name}</b>{name !== doc.file && <> <span className="muted">({doc.file})</span></>}.</p>
          <p className="muted small">File PDF và thông tin trích dẫn sẽ bị xóa khỏi máy chủ, sau đó index được dựng lại
            (khoảng 1 phút). Thao tác này không hoàn tác được.</p>
        </>
      ),
      confirmLabel: "Xóa tài liệu",
      danger: true,
    });
    if (!ok) return;
    setDeleting(doc.file);
    try {
      await api.deleteDocument(doc.file);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setDeleting(null);
    }
  };

  const saveMeta = async () => {
    try {
      await api.updateMetadata(editing.file, editing.meta);
      setEditing(null);
      await refresh();
    } catch (e) {
      setError(e.message);
    }
  };

  const rebuild = async () => {
    try {
      const job = await api.rebuild();
      setData((d) => ({ ...d, job }));
    } catch (e) {
      setError(e.message);
    }
  };

  const pendingCount = queue.filter((it) => it.status === "pending" || it.status === "error").length;

  return (
    <section>
      {confirmDialog}
      {error && <div className="alert alert-error">{error}</div>}
      {data && <JobStatus job={data.job} needsRebuild={data.needs_rebuild} onRebuild={rebuild} busy={busy} />}

      {/* ---------- Khung upload ---------- */}
      <div className={dragging ? "dropzone dragging" : "dropzone"}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}>
        <input ref={inputRef} type="file" accept=".pdf,application/pdf" multiple hidden
          onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
        <div className="dropzone-icon">📄</div>
        <b>Kéo thả file PDF vào đây hoặc bấm để chọn</b>
        <span className="muted small">Có thể chọn nhiều file · tối đa {data?.max_upload_mb ?? 50} MB/file · PDF phải có lớp chữ (không phải ảnh scan)</span>
      </div>

      {queue.length > 0 && (
        <div className="queue">
          {queue.map((it) => (
            <div key={it.id} className={`queue-item ${it.status}`}>
              <div className="queue-row">
                <span className="queue-name" title={it.file.name}>{it.file.name}</span>
                <span className="muted small">{fmtSize(it.file.size / 1024)}</span>
                <span className="queue-status small">
                  {it.status === "uploading" && <span className="spinner sm" aria-hidden="true" />}
                  {{ pending: "Chờ tải lên", uploading: "Đang tải...", done: "✓ Đã tải", error: "✗ Lỗi" }[it.status]}
                </span>
                {it.status !== "uploading" && it.status !== "done" && (
                  <>
                    <button className="link-btn" onClick={() => updateItem(it.id, { open: !it.open })}>
                      {it.open ? "Ẩn thông tin" : "Thông tin trích dẫn"}
                    </button>
                    <button className="link-btn danger" onClick={() => setQueue((q) => q.filter((x) => x.id !== it.id))}>
                      Bỏ
                    </button>
                  </>
                )}
              </div>
              {it.message && <p className={it.status === "error" ? "small text-error" : "small muted"}>{it.message}</p>}
              {it.open && <MetaForm value={it.meta} onChange={(meta) => updateItem(it.id, { meta })} />}
            </div>
          ))}
          <div className="queue-actions">
            <label className="check small">
              <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />
              Ghi đè nếu trùng tên
            </label>
            <button className="btn" onClick={() => setQueue([])} disabled={uploading}>Xóa danh sách</button>
            <button className="btn primary" onClick={uploadAll} disabled={busy || !pendingCount}>
              {uploading ? <><span className="spinner sm" aria-hidden="true" /> Đang tải lên...</>
                : `Tải lên ${pendingCount} file & dựng index`}
            </button>
          </div>
          <p className="muted small">
            "Thông tin trích dẫn" là tùy chọn: nếu bỏ trống, tên file sẽ được dùng làm nguồn trong câu trả lời.
          </p>
        </div>
      )}

      {/* ---------- Danh sách tài liệu ---------- */}
      <h3>
        Tài liệu hiện có {data && <span className="muted">({data.documents.length})</span>}
        {data?.index_built_at && <span className="muted small"> · index dựng lúc {fmtDateTime(data.index_built_at)}</span>}
      </h3>
      {!data ? <p className="muted">Đang tải...</p> : data.documents.length === 0 ? (
        <p className="muted">Chưa có tài liệu nào. Hãy tải PDF lên ở khung phía trên.</p>
      ) : (
        <div className="doc-list">
          {data.documents.map((doc) => {
            const m = doc.metadata;
            const isEditing = editing?.file === doc.file;
            return (
              <div key={doc.file} className="doc-item">
                <div className="doc-row">
                  <div className="doc-main">
                    <b>{m.short_ref || m.title || doc.file}</b>
                    {m.title && m.short_ref && <span className="small">{m.title}</span>}
                    <span className="muted small">{doc.file} · {fmtSize(doc.size_kb)}</span>
                  </div>
                  <span className={doc.indexed ? "status ok" : "status warn"}>
                    {doc.indexed ? "Đã index" : "Chưa index"}
                  </span>
                  <div className="doc-actions">
                    <button className="btn small" disabled={busy}
                      onClick={() => setEditing(isEditing ? null : { file: doc.file, meta: { ...m } })}>
                      {isEditing ? "Đóng" : "Sửa thông tin"}
                    </button>
                    <button className="btn small danger" disabled={busy || deleting !== null} onClick={() => remove(doc)}>
                      {deleting === doc.file ? <><span className="spinner sm" aria-hidden="true" />Đang xóa</> : "Xóa"}
                    </button>
                  </div>
                </div>
                {isEditing && (
                  <div className="doc-edit">
                    <MetaForm value={editing.meta} onChange={(meta) => setEditing({ ...editing, meta })} />
                    <div className="queue-actions">
                      <span className="muted small">Sau khi lưu cần dựng lại index để trích dẫn được cập nhật.</span>
                      <button className="btn primary" onClick={saveMeta}>Lưu</button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
