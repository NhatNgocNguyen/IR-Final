# Checklist & Xử lý sự cố ngày thi

Tài liệu cầm tay cho 15 phút setup Student RAG Server ở phòng thi.
Máy: Windows + PowerShell. Python 3.11, model trên Git LFS (~515MB).

---

## 0. Thứ tự "vàng" trong 15 phút

```powershell
# 1. Lấy code + model
git clone --depth 1 <repo_url>
cd <repo>
git lfs install; git lfs pull
(Get-Item models\vietnamese-sbert\model.safetensors).Length   # phải ra ~540000000

# 2. Môi trường (gọi thẳng python trong venv, khỏi lo execution policy)
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Cấu hình
Copy-Item .env.example .env
ipconfig    # lấy IP LAN (192.168.x.x), điền SERVER_PUBLIC_IP + STUDENT_ID vào .env

# 4. Chạy + kiểm tra
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# tab PowerShell khác:
curl http://127.0.0.1:8000/health      # chờ rag_ready=true, index_persisted=true
```

---

## 1. Về lệnh `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Nguyên nhân:** PowerShell mặc định thường ở chế độ `Restricted` (cấm chạy file `.ps1`).
File kích hoạt venv `Activate.ps1` là một script `.ps1`, nên bị chặn với lỗi:

> Activate.ps1 cannot be loaded because running scripts is disabled on this system.

**Lệnh xử lý:**

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

- `RemoteSigned`: script tạo trên máy chạy tự do; script tải từ mạng mới cần chữ ký.
- `-Scope CurrentUser`: chỉ cho tài khoản hiện tại, **không cần quyền admin**, an toàn.

**Mẹo né hẳn:** đừng activate venv. Gọi thẳng python trong venv:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Cách này không chạy file `.ps1` nào nên bỏ qua hoàn toàn execution policy.

---

## 2. Giai đoạn A — Clone code & lấy model

| Sự cố | Dấu hiệu | Lệnh xử lý |
|---|---|---|
| Git LFS chưa cài → model là file giả | `model.safetensors` chỉ ~130 byte; load model lỗi | `git lfs install` rồi `git lfs pull` |
| Kiểm tra model có thật không | Cần xác nhận sau clone | `(Get-Item models\vietnamese-sbert\model.safetensors).Length` → phải ~**540000000**, không phải ~130 |
| Clone vào thư mục không có quyền ghi | Lỗi permission khi tạo venv / ghi `storage/index` | Clone vào `Desktop` hoặc `Downloads`, tránh `C:\Program Files` |

---

## 3. Giai đoạn B — Tạo venv & cài thư viện

| Sự cố | Dấu hiệu | Lệnh xử lý |
|---|---|---|
| Execution policy chặn activate | "running scripts is disabled" | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` hoặc gọi thẳng `.\.venv\Scripts\python.exe` |
| Không có Python 3.11 | `py -3.11` không tìm thấy | Liệt kê bản có: `py -0p`. Dùng tạm: `py -3.12 -m venv .venv` |
| `py` launcher không có | `py` không nhận | Dùng `python -m venv .venv` |
| pip cài chậm / kéo nhầm bản torch CUDA | Tải rất lâu | Cài torch CPU trước: `pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu` rồi `pip install -r requirements.txt` |
| pip timeout do mạng | Đứt giữa chừng | Thêm `--timeout 120 --retries 5`; chạy lại để resume |

---

## 4. Giai đoạn C — Cấu hình `.env`

| Sự cố | Dấu hiệu | Lệnh xử lý |
|---|---|---|
| Thiếu `.env` | Crash: "Missing required environment variable: STUDENT_ID" | `Copy-Item .env.example .env` rồi mở sửa |
| STUDENT_ID sai | Teacher không nhận | Điền đúng **MSSV viết HOA** |
| IP teacher khác hôm thi | Register/evaluate fail | Sửa `TEACHER_BASE_URL` và `TEACHER_PROXY_BASE_URL` theo IP teacher thông báo tại phòng |

---

## 5. Giai đoạn D — Chạy server & mạng LAN (dễ sai nhất)

| Sự cố | Dấu hiệu | Lệnh xử lý |
|---|---|---|
| Đăng ký nhầm IP (localhost / VPN / nhiều card mạng) | Teacher gọi `/upload` `/ask` không tới | Xem IP LAN: `ipconfig` (IPv4 `192.168.x.x`). Đặt `SERVER_PUBLIC_IP=192.168.x.x` trong `.env` |
| Firewall chặn cổng 8000 | Teacher không kết nối dù server chạy | Khi uvicorn khởi động, Windows hiện popup → bấm **Allow access** (cả Private). Hoặc admin: `New-NetFirewallRule -DisplayName "RAG8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow` |
| Cổng 8000 đã bị chiếm | uvicorn báo "address already in use" | `Get-NetTCPConnection -LocalPort 8000` rồi `Stop-Process -Id <pid>`; hoặc đổi `SERVER_PORT` |
| Quên health check trước khi nộp | Nộp khi index chưa sẵn sàng | `curl http://127.0.0.1:8000/health` → chờ `rag_ready=true` và `index_persisted=true` |
| `/upload` lần đầu chậm (CPU encode nhiều chunk) | Gần chạm timeout 120s | Bình thường vẫn kịp; để server chạy sẵn trước khi teacher gọi |

---

## 6. Giai đoạn E — Tự kiểm tra từ máy khác

| Việc | Lệnh |
|---|---|
| Xác nhận server cho phép truy cập từ ngoài | Từ máy khác / điện thoại cùng wifi: `curl http://<IP_LAN_máy_bạn>:8000/health` |

---

## 7. Helper scripts (sau khi server chạy)

```powershell
.\.venv\Scripts\python.exe scripts/register.py    # đăng ký server lên teacher
.\.venv\Scripts\python.exe scripts/evaluate.py --document-received false   # lần đầu: teacher gửi tài liệu
.\.venv\Scripts\python.exe scripts/evaluate.py --document-received true    # các lần sau: chấm thật
.\.venv\Scripts\python.exe scripts/result.py      # xem điểm/trạng thái
.\.venv\Scripts\python.exe scripts/reset.py        # reset trạng thái trên teacher
```

Sau lần upload đầu: kiểm tra `storage/index` đã có `metadata.json` và `embeddings.npy`.
