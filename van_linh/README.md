# Student RAG Server

FastAPI server cho bài thi RAG offline với 2 endpoint bắt buộc:
- `POST /upload`
- `POST /ask`

Server dùng:
- embedding local `keepitreal/vietnamese-sbert`
- retrieval trong RAM bằng `numpy`
- teacher proxy theo chuẩn OpenAI-compatible API
- local persistence cho retrieval index tại `storage/index`

## 1. Chuẩn bị môi trường

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Cấu hình

Tạo file `.env` từ `.env.example` rồi điền:
- `STUDENT_ID`
- `EMBEDDING_MODEL_PATH`
- `INDEX_STORAGE_DIR`
- `SERVER_PORT`
- nếu cần, `SERVER_PUBLIC_IP`

`SERVER_PUBLIC_IP` dùng để override IP tự dò khi máy có nhiều card mạng hoặc VPN.

## 3. Chạy server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

`/health` sẽ trả thêm:
- `rag_ready`
- `doc_id`
- `chunk_count`
- `index_persisted`

## 4. Quy trình thi mới

Lần đầu khi vào thi:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --document-received false
```

Khi teacher server gửi tài liệu sang `/upload`, server sẽ:
- chunking văn bản
- tạo embeddings
- lưu index local vào `storage/index`

Sau khi upload xong, kiểm tra:

```powershell
curl http://127.0.0.1:8000/health
```

Chỉ tiếp tục nộp lại khi:
- `rag_ready=true`
- `index_persisted=true`

Những lần evaluate sau:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --document-received true
```

Nếu tắt server hoặc restart máy, server sẽ tự load lại index từ `storage/index` khi khởi động lại.

## 5. Gọi helper scripts

Đăng ký server lên teacher server:

```powershell
.\.venv\Scripts\python.exe scripts/register.py
```

Kiểm tra trạng thái:

```powershell
.\.venv\Scripts\python.exe scripts/result.py
```

Reset trạng thái trên teacher server:

```powershell
.\.venv\Scripts\python.exe scripts/reset.py
```

Lưu ý: `register.py` và `reset.py` mặc định không xóa index local.

## 6. Lưu ý về persistence

Persistence chỉ hợp lệ khi các cấu hình sau không đổi:
- `EMBEDDING_MODEL_PATH`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

Nếu thay đổi một trong các cấu hình trên, server sẽ bỏ qua index cũ và cần upload/build lại.

## 7. Checklist trước khi thi

- xác nhận model local load được khi ngắt mạng
- chạy server bằng IP LAN mà teacher server truy cập được
- không đăng ký `localhost` hoặc `127.0.0.1`
- `POST /ask` luôn trả đúng một ký tự `A/B/C/D`
- nếu tự dò IP sai thì set `SERVER_PUBLIC_IP` trong `.env`
- sau lần upload đầu tiên, kiểm tra `storage/index` đã có `metadata.json` và `embeddings.npy`

## 8. Chế độ nạp embedding model (mới)

Server tự chọn nguồn model theo thứ tự ưu tiên, không cần can thiệp:

1. **Folder local trong dự án** — nếu `EMBEDDING_MODEL_PATH` (mặc định `models/vietnamese-sbert`) tồn tại thì dùng folder này với `local_files_only=True` (chạy hoàn toàn offline).
2. **Hugging Face cache** — nếu folder trên không có, server fallback sang model id `EMBEDDING_MODEL_NAME` (mặc định `keepitreal/vietnamese-sbert`), nạp từ HF cache đã tải sẵn.

Hai biến liên quan trong `.env`:

```env
EMBEDDING_MODEL_PATH=models/vietnamese-sbert
EMBEDDING_MODEL_NAME=keepitreal/vietnamese-sbert
```

Để dùng được chế độ fallback khi máy thi chưa có folder model, hãy tải model về HF cache **một lần khi còn mạng**:

```powershell
.\.venv\Scripts\python.exe nhat/download_model.py
```

Lệnh này tải `keepitreal/vietnamese-sbert` về `~/.cache/huggingface/hub`. Sau đó dù dự án không có folder `models/vietnamese-sbert`, server vẫn chạy được.

Log lúc khởi động cho biết đang dùng nguồn nào:

```
Embedding model loaded from <path-or-id> (local_only=True/False)
```

## 9. Chọn mẫu prompt (mới)

Có sẵn 4 mẫu prompt, chọn qua biến số `PROMPT_VARIANT` trong `.env`:

```env
# 1=A can bang, 2=B tai lieu+kien thuc, 3=C cau phu dinh, 4=D toi gian
PROMPT_VARIANT=2
```

| Giá trị | Mẫu | Đặc điểm |
|---------|-----|----------|
| `1` | A | Cân bằng, nhận diện câu đúng/sai |
| `2` | B (mặc định) | Tài liệu + kiến thức nền, hợp câu suy luận |
| `3` | C | Tập trung câu phủ định / ngoại lệ |
| `4` | D | Tối giản, output ngắn |

Đổi prompt **không cần re-index**, chỉ cần restart server. Giá trị ngoài khoảng 1-4 sẽ tự fallback về `2`. Chi tiết nội dung từng mẫu xem [PROMPT_SAMPLES.md](PROMPT_SAMPLES.md).

Mỗi `/ask` ghi log mẫu đang dùng:

```
Teacher proxy returned answer=B (prompt_variant=2, ~1800 tokens, chunks=7/7, budget=4000) in 0.8s
```

## 10. Giới hạn token mỗi request (mới)

Để tuân thủ giới hạn 4K token/request của teacher và tránh bị bóp băng thông, server tự cắt context cho vừa ngân sách trước khi gọi LLM:

```env
MAX_PROMPT_TOKENS=4000      # tran token toi da moi request
PROMPT_TOKEN_MARGIN=300     # du phong output + overhead chat
GPT_CHARS_PER_TOKEN=2.5     # uoc luong token tu so ky tu (conservative)
LLM_MAX_RETRIES=1           # giam retry de tranh spam request
```

Cách hoạt động:
- Ngân sách context = `MAX_PROMPT_TOKENS - PROMPT_TOKEN_MARGIN`.
- Các chunk được gom theo thứ tự retrieval cho tới khi gần chạm ngân sách; phần dư bị bỏ.
- Token được ước lượng từ số ký tự (không cần thư viện `tiktoken` lúc chạy), dùng tỉ lệ conservative để luôn ước lượng dư, tránh vượt 4K thật.
- Mỗi câu chỉ gọi LLM **một lần**; `LLM_MAX_RETRIES=1` hạn chế gọi lặp khi mạng chập chờn.

Muốn chặt hơn: tăng `PROMPT_TOKEN_MARGIN` hoặc giảm `GPT_CHARS_PER_TOKEN`.

## 11. Lưu ý persistence bổ sung (mới)

Ngoài `EMBEDDING_MODEL_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP` ở mục 6, index local còn bị bỏ và phải build lại khi đổi:
- `MAX_CHUNK_TOKENS` — đổi cách cắt chunk theo token.
- nguồn embedding model (đường dẫn local hoặc model id) — phải khớp với giá trị đã lưu trong `metadata.json`.

Index cũng tự động bỏ khi `retrieval_version` trong code thay đổi (đảm bảo không dùng nhầm index sinh từ logic cũ).
