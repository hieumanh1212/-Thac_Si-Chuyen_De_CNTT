// Lớp gọi backend. Mọi giao tiếp FE <-> BE đều đi qua file này.
const BASE = import.meta.env.VITE_API_URL || ""; // rỗng => dùng proxy /api của Vite

async function request(path, options = {}) {
  const isForm = options.body instanceof FormData; // upload file: để trình duyệt tự đặt Content-Type
  let res;
  try {
    res = await fetch(BASE + path, {
      ...options,
      headers: isForm || !options.body ? {} : { "Content-Type": "application/json" },
    });
  } catch {
    throw new Error("Không kết nối được backend. Hãy kiểm tra server FastAPI đã chạy ở cổng 8000 chưa.");
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* phản hồi không phải JSON */
  }
  if (!res.ok) {
    const detail = data?.detail;
    const msg = typeof detail === "string" ? detail
      : Array.isArray(detail) ? detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ")
      : `HTTP ${res.status}`;
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

const send = (method, path, body) => request(path, { method, body: JSON.stringify(body) });
const enc = encodeURIComponent;

export const api = {
  info: () => request("/api/info"),
  stats: () => request("/api/stats"),
  ask: (body) => send("POST", "/api/ask", body),
  compare: (body) => send("POST", "/api/compare", body),

  // ---- quản lý tài liệu ----
  documents: () => request("/api/documents"),
  upload: (file, metadata = {}, { overwrite = false, rebuild = true } = {}) => {
    const form = new FormData();
    form.append("file", file);
    form.append("metadata", JSON.stringify(metadata));
    form.append("overwrite", String(overwrite));
    form.append("rebuild", String(rebuild));
    return request("/api/documents", { method: "POST", body: form });
  },
  updateMetadata: (file, metadata) => send("PUT", `/api/documents/${enc(file)}/metadata`, metadata),
  deleteDocument: (file, rebuild = true) =>
    request(`/api/documents/${enc(file)}?rebuild=${rebuild}`, { method: "DELETE" }),
  rebuild: () => request("/api/index/rebuild", { method: "POST" }),
  indexStatus: () => request("/api/index/status"),
};

// Chuyển state cấu hình ở sidebar thành tham số tìm kiếm của API
export function searchParams(s) {
  return {
    strategy: s.strategy,
    mode: s.mode,
    top_k: s.topK,
    bm25_top_n: s.bm25TopN,
    faiss_top_n: s.faissTopN,
    rrf_k: s.rrfK,
    sources: s.sources.length ? s.sources : null,
  };
}
