# Student RAG Server

FastAPI server cho bài thi RAG offline với 2 endpoint bắt buộc:
- `POST /upload`
- `POST /ask`

Server dùng:
- embedding local `keepitreal/vietnamese-sbert`
- retrieval trong RAM bằng `numpy`
- teacher proxy theo chuẩn OpenAI-compatible API

## 1. Chuẩn bị môi trường

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Cấu hình

Tạo file `.env` từ `.env.example` rồi điền:
- `STUDENT_ID`
- `EMBEDDING_MODEL_PATH`
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

## 4. Gọi helper scripts

Đăng ký server lên teacher server:

```powershell
.\.venv\Scripts\python.exe scripts/register.py
```

Bắt đầu chấm:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py
```

Kiểm tra trạng thái:

```powershell
.\.venv\Scripts\python.exe scripts/result.py
```

Reset trạng thái:

```powershell
.\.venv\Scripts\python.exe scripts/reset.py
```

## 5. Checklist trước khi thi

- xác nhận model local load được khi ngắt mạng
- chạy server bằng IP LAN mà teacher server truy cập được
- không đăng ký `localhost` hoặc `127.0.0.1`
- `POST /ask` luôn trả đúng một ký tự `A/B/C/D`
- nếu tự dò IP sai thì set `SERVER_PUBLIC_IP` trong `.env`
