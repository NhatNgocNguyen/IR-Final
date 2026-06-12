# Student RAG Server

FastAPI server cho bài thi RAG offline (PTIT). Server nhận tài liệu, lập chỉ mục vector, rồi trả lời câu hỏi trắc nghiệm bằng **một ký tự** `A`, `B`, `C` hoặc `D`.

## Kiến trúc

```
/upload  →  chunker → embedder → vectordb (BM25 + dense)
/ask     →  retriever → LLM proxy (gpt-4o-mini) → A/B/C/D
```

| Thành phần | File | Mô tả |
|---|---|---|
| API server | `main.py` | Endpoints `/upload`, `/ask`, `/health` |
| RAG pipeline | `rag_pipeline.py` | Điều phối upload + trả lời |
| Chunking | `chunker.py` | Chia văn bản tiếng Việt theo đoạn/câu |
| Embedding | `embedder.py` | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | `vectordb.py` | Hybrid BM25 + cosine similarity |
| Retrieval | `retriever.py` | Tìm kiếm, de-duplicate, lắp context |
| LLM client | `llm_client.py` | Gọi teacher proxy (OpenAI-compatible) |
| Đăng ký thi | `register.py` | CLI gọi Teacher Server |
| Cấu hình | `config.py` | Toàn bộ tham số tập trung một chỗ |

## 1. Chuẩn bị môi trường

Yêu cầu: Python 3.10+ (khuyến nghị 3.11).

```bash
cd nhat
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## 2. Tải model embedding (chạy một lần khi còn internet)

Model ~420 MB, được cache vào thư mục Hugging Face local để thi offline.

```bash
python download_model.py
```

## 3. Cấu hình trước khi thi

Mở `config.py` và sửa **2 dòng bắt buộc**:

```python
STUDENT_ID    = "YOUR_MSSV_HERE"          # mã sinh viên
MY_SERVER_URL = "http://192.168.1.X:5000" # IP LAN + port của máy bạn
```

Hoặc dùng biến môi trường:

```bash
export STUDENT_ID="B20XXXXX"
export MY_SERVER_URL="http://192.168.1.100:5000"
export PORT=5000   # tùy chọn, mặc định 5000
```

**Lưu ý:** `MY_SERVER_URL` phải là địa chỉ LAN mà Teacher Server truy cập được. Không dùng `localhost` hoặc `127.0.0.1`.

## 4. Chạy server

```bash
python main.py
```

Server lắng nghe tại `0.0.0.0:5000` (hoặc port trong `PORT`).

Kiểm tra trạng thái:

```bash
curl http://127.0.0.1:5000/health
```

Kết quả mẫu:

```json
{
  "status": "ok",
  "vectordb_ready": false,
  "chunks": 0,
  "doc_id": null
}
```

Sau khi Teacher gửi tài liệu qua `/upload`, `vectordb_ready` sẽ là `true`.

## 5. API endpoints

### `POST /upload`

Nhận tài liệu, chunk + embed + lưu vào VectorDB.

```bash
curl -X POST http://127.0.0.1:5000/upload \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "exam_doc", "text": "Nội dung tài liệu..."}'
```

Response:

```json
{"status": "success", "doc_id": "exam_doc", "chunks": 42}
```

### `POST /ask`

Nhận câu hỏi trắc nghiệm, trả về **đúng một chữ cái**.

```bash
curl -X POST http://127.0.0.1:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG là viết tắt của gì?\nA. ...\nB. ...\nC. ...\nD. ..."}'
```

Response:

```json
{"answer": "B", "sources": ["đoạn văn trích dẫn..."]}
```

### `GET /health`

Xem trạng thái VectorDB (dùng khi debug trong lúc thi).

## 6. Đăng ký và nộp bài (Teacher Server)

Dùng `register.py` để tương tác với Teacher Server (`192.168.50.218:8000`):

```bash
# Đăng ký URL server của bạn
python register.py register

# Bắt đầu chấm (upload tài liệu + 100 câu hỏi)
python register.py evaluate

# Bỏ qua upload nếu VectorDB đã có sẵn (nộp lại nhanh hơn)
python register.py evaluate --skip-upload

# Xem điểm hiện tại
python register.py result

# Reset trạng thái (dùng 1 trong 5 lần nộp)
python register.py reset

# Tiện lợi: register + evaluate một lệnh
python register.py start
python register.py start --skip-upload
```

## 7. Luồng thi điển hình

1. Có internet → `pip install` + `python download_model.py`
2. Sửa `STUDENT_ID` và `MY_SERVER_URL` trong `config.py`
3. Ngắt internet (nếu thi offline) → `python main.py`
4. Xác nhận `/health` trả `status: ok`
5. `python register.py start`
6. Chờ Teacher gửi `/upload` rồi 100 lần `/ask` (~ vài phút)
7. `python register.py result` để xem điểm

## 8. Dữ liệu lưu trữ

VectorDB được tự động lưu tại `data/vectordb.pkl` sau mỗi lần `/upload`. Khi khởi động lại server, index được khôi phục — không cần upload lại nếu dùng `--skip-upload`.

## 9. Tinh chỉnh (tùy chọn)

Các tham số quan trọng trong `config.py`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `CHUNK_SIZE` | 450 | Kích thước mỗi chunk (ký tự) |
| `CHUNK_OVERLAP` | 90 | Overlap giữa các chunk |
| `TOP_K_FINAL` | 6 | Số chunk gửi cho LLM |
| `BM25_WEIGHT` / `DENSE_WEIGHT` | 0.40 / 0.60 | Trọng số hybrid search |
| `MAX_CONTEXT_CHARS` | ~3100 | Giới hạn context (teacher max 4000 chars/prompt) |
| `LLM_TEMPERATURE` | 0.0 | Deterministic — câu trả lời ổn định |

## 10. Checklist trước khi thi

- [ ] Model embedding đã tải xong (`download_model.py` chạy thành công)
- [ ] Server chạy được khi **không có internet**
- [ ] `MY_SERVER_URL` trỏ đúng IP LAN, Teacher ping được
- [ ] `STUDENT_ID` đúng MSSV
- [ ] `/ask` luôn trả đúng một ký tự `A` / `B` / `C` / `D`
- [ ] Firewall cho phép kết nối đến port 5000 từ mạng LAN thi

## Xử lý sự cố

| Vấn đề | Cách xử lý |
|---|---|
| `VectorDB not ready` khi `/ask` | Chưa có `/upload` — chờ Teacher hoặc test thủ công |
| Model load lỗi | Chạy lại `python download_model.py` khi có mạng |
| Teacher không kết nối được | Kiểm tra IP LAN, firewall, cùng subnet với `192.168.50.218` |
| Điểm thấp khi nộp lại | Thử chỉnh `TOP_K_FINAL`, `BM25_WEIGHT`, hoặc `CHUNK_SIZE` rồi reset + nộp lại |
