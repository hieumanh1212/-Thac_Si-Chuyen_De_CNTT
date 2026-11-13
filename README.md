# Hybrid RAG với các chiến lược Chunking

Ứng dụng hỏi đáp trên tài liệu PDF: tìm đoạn liên quan bằng **BM25 + FAISS hợp nhất qua RRF**,
sau đó LLM trả lời **chỉ dựa trên tài liệu**, kèm trích dẫn nguồn / mục / trang.
So sánh 3 chiến lược chunking: **Fixed-size**, **Recursive**, **Semantic**.

```
PDF ─► đọc từng trang ─► làm sạch ─► chunking ─┬─► BM25  Top-8 ─┐
                                               └─► FAISS Top-8 ─┴─► RRF ─► Top-K ─► Context ─► LLM ─► Trả lời + citation
```

## Kiến trúc: Backend và Frontend tách riêng

```
┌──────────────────────┐   HTTP / JSON    ┌───────────────────────────┐
│  frontend/ (React)   │ ───────────────► │  backend/ (FastAPI)       │
│  Vite, cổng 5173     │   /api/ask ...   │  cổng 8000                │
│  chỉ lo hiển thị     │ ◄─────────────── │  api/  ─►  rag/ (pipeline)│
└──────────────────────┘                  └───────────────────────────┘
```

- **backend/**: toàn bộ xử lý RAG (đọc PDF, chunking, index, tìm kiếm, LLM, đánh giá) + REST API.
  Chạy được độc lập: dựng index, đánh giá, test đều không cần frontend.
- **frontend/**: giao diện React. Chỉ giao tiếp với backend qua API, không chứa logic RAG.

| API | Mô tả |
|---|---|
| `GET /api/health` | Kiểm tra server sống |
| `GET /api/info` | Chiến lược đã có index, chế độ tìm kiếm, tham số mặc định, danh sách tài liệu |
| `POST /api/ask` | Hỏi đáp: truy xuất + LLM, trả về câu trả lời có citation và Top-K đoạn |
| `POST /api/compare` | Cùng câu hỏi, so sánh kết quả truy xuất giữa các chiến lược |
| `GET /api/stats` | Thống kê index (số chunk, độ dài, phân bố độ dài...) |
| `GET /api/documents` | Danh sách PDF, trạng thái đã/chưa index, có cần dựng lại index không |
| `POST /api/documents` | Upload 1 PDF (multipart) kèm thông tin trích dẫn tùy chọn |
| `PUT /api/documents/{file}/metadata` | Sửa thông tin trích dẫn (tiêu đề, tác giả, năm, short_ref...) |
| `DELETE /api/documents/{file}` | Xóa PDF |
| `POST /api/index/rebuild` · `GET /api/index/status` | Dựng lại index (chạy nền) và xem tiến độ |

Tài liệu API tương tác (bấm thử trực tiếp): http://localhost:8000/docs

## 1. Cài đặt

Cần **Python 3.10 – 3.12** và **Node.js 18+** (tải tại nodejs.org).

```bash
# Backend (môi trường ảo .venv đặt ở thư mục gốc hybrid-rag/)
cd hybrid-rag
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt

# Frontend
cd frontend
npm install
```

Không dùng `pip install -e .` vì cách cài này lỗi khi đường dẫn có dấu tiếng Việt trên Windows.
Lần chạy đầu tiên sẽ tự tải mô hình embedding (~1 GB) từ HuggingFace.

## 2. Cấu hình API key cho LLM

```bash
cd backend
copy .env.example .env                 # macOS/Linux: cp .env.example .env
```

Điền key của nhà cung cấp LLM (Claude, OpenAI, Gemini hoặc Ollama local) vào `backend/.env`.
File `.env` nằm trong `.gitignore`, **không commit key lên git**. Phần tìm kiếm và đánh giá
retrieval **không cần API key**; nếu chưa có key, giao diện vẫn hiện các đoạn truy xuất.

## 3. Chuẩn bị dữ liệu và dựng index

**Cách 1 – qua giao diện (khuyên dùng):** mở tab **📁 Tài liệu**, kéo thả PDF vào khung upload,
điền "Thông tin trích dẫn" (tùy chọn) rồi bấm *Tải lên & dựng index*. Hệ thống tự dựng lại index
cho cả 3 chiến lược (khoảng 1 phút với vài tài liệu); trong lúc chờ vẫn hỏi đáp được bằng index cũ.
Thông tin nhập trên giao diện được lưu ở `backend/data/sources_uploaded.yaml`.

**Cách 2 – thủ công:**

1. Chép PDF vào `backend/data/raw/`.
2. Khai báo metadata trong `backend/data/sources.yaml` (`short_ref` dùng làm tên nguồn trong citation).
3. Dựng index (chạy trong thư mục `backend/`, đã activate `.venv`):

```bash
python -m rag.indexing.build_index --strategy all
```

## 4. Chạy ứng dụng

**Cách nhanh (Windows):** bấm đúp `start.bat` ở thư mục gốc. File này mở 2 cửa sổ Backend + Frontend rồi tự mở trình duyệt tại http://localhost:5173. Muốn tắt thì đóng 2 cửa sổ đó.

**Cách thủ công (mở 2 terminal):**

**Terminal 1 – Backend**
```bash
cd hybrid-rag
.venv\Scripts\activate
cd backend
uvicorn api.main:app --port 8000 --reload
```
Chờ đến khi thấy `Application startup complete`.

**Terminal 2 – Frontend**
```bash
cd hybrid-rag/frontend
npm run dev
```

Mở trình duyệt: **http://localhost:5173**

Vite tự chuyển tiếp mọi request `/api/...` sang `http://localhost:8000`, nên không cần cấu hình gì thêm.
Muốn trỏ sang backend khác: tạo `frontend/.env` với `BACKEND_URL=http://...` (xem `frontend/.env.example`).

## 5. Đánh giá và kiểm thử (trong `backend/`)

```bash
python -m rag.evaluation.run_eval                # đánh giá retrieval trên eval/questions.csv
python -m rag.evaluation.run_eval --with-llm     # thêm câu trả lời LLM để chấm tay
pytest                                           # unit test (gồm cả test API)
```

### Bộ câu hỏi đánh giá (`backend/eval/questions.csv`)

| Cột | Ý nghĩa |
|---|---|
| `id` | Mã câu hỏi (Q01...) |
| `query` | Câu hỏi |
| `type` | `keyword` (khớp từ khóa) / `semantic` (diễn đạt lại) / `synthesis` (tổng hợp nhiều nguồn) |
| `ground_truth` | Vị trí đáp án: `file.pdf:trang`, nhiều nguồn cách nhau bằng `;`. Ví dụ `a.pdf:3-4; b.pdf:7` |
| `gt_section` | Mục chứa đáp án (để đối chiếu trong báo cáo) |

Số trang là **số trang trong file PDF** (trang trình xem PDF hiển thị), không phải số in trên giấy.
Kết quả lưu ở `backend/eval/results/eval_<thời gian>.xlsx` (sheet `per_query`, `summary`,
`summary_by_type`, `answers`). Metrics: Precision@3, Recall@3, Hit@3, MRR, nDCG@3, thời gian, token context.

## 6. Cấu trúc thư mục

```
hybrid-rag/
├── .venv/                           # môi trường ảo Python (không commit)
├── backend/
│   ├── api/
│   │   ├── main.py                  # FastAPI: hỏi đáp, so sánh, thống kê
│   │   ├── documents.py             # upload/xóa PDF, sửa metadata, dựng index chạy nền
│   │   └── schemas.py               # định dạng request/response (hợp đồng với frontend)
│   ├── rag/                         # package RAG
│   │   ├── config.py                # nạp config + .env
│   │   ├── schemas.py               # Document, Chunk, RetrievedChunk
│   │   ├── ingest/                  # đọc PDF từng trang, làm sạch, nhận diện mục
│   │   ├── chunking/                # fixed_size.py, recursive.py, semantic.py
│   │   ├── indexing/                # embedder, faiss_store, bm25_store, build_index
│   │   ├── retrieval/               # rrf.py, hybrid.py
│   │   ├── generation/              # prompts, context_builder (citation), llm_client
│   │   ├── evaluation/              # metrics.py, run_eval.py
│   │   └── pipeline.py              # RAGPipeline.ask(): điểm vào chung cho API/eval/notebook
│   ├── configs/config.yaml          # MỌI tham số: chunk_size, overlap, top-k, rrf_k, model...
│   ├── data/
│   │   ├── raw/                     # PDF gốc
│   │   ├── sources.yaml             # metadata cấp tài liệu
│   │   ├── processed/               # documents.jsonl
│   │   └── indexes/<strategy>/      # chunks.jsonl, faiss.npy, bm25.pkl, meta.json
│   ├── eval/questions.csv           # bộ câu hỏi đánh giá
│   ├── tests/                       # unit test
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── api.js                   # mọi lời gọi tới backend
    │   ├── App.jsx                  # khung trang + tab
    │   └── components/              # Sidebar, AskTab, CompareTab, StatsTab, DocumentsTab, ChunkCard
    ├── vite.config.js               # proxy /api -> backend
    └── package.json
```

## 7. Thiết kế chính

- **Giữ metadata trang**: PDF được đọc theo từng trang; vị trí ký tự của từng trang được lưu lại,
  nên chunk vắt qua 2 trang vẫn biết chính xác `page` và `end_page`.
- **Section**: lấy từ mục lục (bookmark) của PDF nếu có; nếu không thì nhận diện tiêu đề bằng heuristic.
- **chunk_id thống nhất**: FAISS dùng `IndexIDMap2` với chính chunk_id của BM25, nên RRF hợp nhất
  đúng cùng một nội dung; khi nạp index chương trình kiểm tra lại sự khớp này.
- **Citation không phụ thuộc LLM**: LLM chỉ trích dẫn `[1]`, `[2]`; code đổi thành
  `[Tác giả, năm, mục, trang]` theo `citation.template`.
- **Ground truth theo trang** (không theo chunk_id) nên một bộ câu hỏi dùng chung cho cả 3 chiến lược.
- **LLM lỗi không làm hỏng request**: API vẫn trả kết quả truy xuất kèm trường `llm_error`.

## 8. Phần cần tùy chỉnh theo đề tài

| File (trong `backend/`) | Việc cần làm |
|---|---|
| `data/sources.yaml` | Metadata đặc thù (Đề 1: course/chapter · Đề 2: số hiệu, ngày ban hành · Đề 3: authors/year/doi) |
| `configs/config.yaml` → `citation.template` | Mẫu citation theo yêu cầu của đề |
| `rag/generation/prompts.py` | Vai trò trợ lý, quy tắc trả lời |
| `rag/ingest/section_detector.py` | Từ khóa / mẫu tiêu đề của loại tài liệu đang dùng |
| `eval/questions.csv` | Tối thiểu 15 câu (5 keyword, 5 semantic, 5 synthesis) |
