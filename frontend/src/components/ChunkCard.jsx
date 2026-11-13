// Một đoạn (chunk) truy xuất được, bấm để mở xem nội dung
export default function ChunkCard({ chunk, index, showRanks = true, compact = false }) {
  const title = compact
    ? `#${chunk.rank} ${chunk.section || chunk.source} · tr.${chunk.pages}`
    : `[${index}] ${chunk.label}`;
  return (
    <details className={chunk.cited ? "chunk cited" : "chunk"}>
      <summary>
        <span className="chunk-mark">{chunk.cited ? "✅" : "▫️"}</span>
        <span className="chunk-title">{title}</span>
        {showRanks && (
          <span className="chunk-ranks">
            BM25 #{chunk.bm25_rank ?? "-"} · FAISS #{chunk.faiss_rank ?? "-"} · score {chunk.score.toFixed(4)}
          </span>
        )}
      </summary>
      <div className="chunk-body">
        <p className="chunk-text">{chunk.text}</p>
        <p className="muted small">
          chunk_id={chunk.chunk_id} · {chunk.text.length} ký tự · strategy={chunk.strategy}
          {chunk.chapter ? ` · chương: ${chunk.chapter}` : ""}
          {compact ? ` · ${chunk.source}` : ""}
        </p>
      </div>
    </details>
  );
}
