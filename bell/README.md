# IR Student Server Project

Dự án này là server FastAPI của sinh viên dùng để nhận tài liệu từ server giáo viên, xử lý truy vấn và gửi kết quả đánh giá.

## 1. Yêu cầu môi trường

Nên dùng Python 3.11.

Kiểm tra Python:

```powershell
py -3.11 --version
```

Nếu chưa có Git, cần cài Git trước rồi clone project:

```powershell
git clone https://github.com/Belldenchoi/IR.git
cd IR
```

Nếu bạn đang dùng thư mục tải sẵn:

```powershell
cd C:\Users\503\Downloads\IR-main
```

## 2. Tạo môi trường ảo

Trong thư mục project, chạy:

```powershell
py -3.11 -m venv .venv
```

Kích hoạt môi trường ảo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell báo lỗi không cho activate, chạy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Cài thư viện

Cài toàn bộ thư viện bằng file `requirements.txt`:

```powershell
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements.txt
```

Kiểm tra nhanh FastAPI và OpenAI:

```powershell
py -3.11 -c "from fastapi import FastAPI; from openai import OpenAI; print('OK')"
```

## 4. Tải model embedding

Project đang dùng model local tại:

```text
models/vietnamese-sbert
```

Nếu chưa có thư mục này, chạy file tải model:

```powershell
py -3.11 download_model.py
```

Sau khi tải xong, cấu trúc thư mục sẽ là:

```text
IR-main/
├── server.py
├── register.py
├── evaluate.py
├── download_model.py
├── requirements.txt
├── README.md
└── models/
    └── vietnamese-sbert/
```

Nếu chưa có `download_model.py`, tạo file này với nội dung:

```python
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "keepitreal/vietnamese-sbert"
SAVE_DIR = Path("models") / "vietnamese-sbert"

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Saving to: {SAVE_DIR.resolve()}")

    model = SentenceTransformer(MODEL_NAME)
    model.save(str(SAVE_DIR))

    print("Done!")
    print(f"Model saved to: {SAVE_DIR.resolve()}")

if __name__ == "__main__":
    main()
```

## 5. Cấu hình IP

Kiểm tra IP máy sinh viên:

```powershell
ipconfig
```

Chọn IP của card mạng thật. Ví dụ:

```text
IPv4 Address . . . . . . . . . . . : 10.170.45.56
```

Không nên dùng các IP của VMware hoặc Loopback như:

```text
192.168.x.x
169.254.x.x
```

Ví dụ cấu hình nên dùng:

```text
Teacher server: http://10.170.45.200:8000
Student server: http://10.170.45.56:5000
```

Trong `register.py`, nên có dạng:

```python
TEACHER_URL = "http://10.170.45.200:8000"
STUDENT_SERVER_URL = "http://10.170.45.56:5000"

url = f"{TEACHER_URL}/register"
```

Lưu ý URL phải là `http://`, không phải `http:/`.

Sai:

```text
http:/10.170.45.200:8000
```

Đúng:

```text
http://10.170.45.200:8000
```

## 6. Cách chạy dự án

Luồng chạy đúng là:

1. Bật student server trước.
2. Chạy `register.py` để đăng ký server với teacher.
3. Chạy `evaluate.py` để bắt đầu đánh giá.

## 7. Bước 1: Bật student server

Mở terminal thứ nhất:

```powershell
cd C:\Users\503\Downloads\IR-main
.\.venv\Scripts\Activate.ps1
py -3.11 -m uvicorn server:app --host 0.0.0.0 --port 5000
```

Mở student server t toàn run python file server.py thôi nên là chạy thế cho nhanh

```powershell
py -3.11 server.py
```

Nếu chạy thành công, terminal sẽ hiện dạng:

```text
Uvicorn running on http://0.0.0.0:5000
```

Không tắt terminal này trong lúc thi/chạy evaluate.

Có thể test server bằng trình duyệt:

```text
http://127.0.0.1:5000/docs
```

## 8. Bước 2: Chạy register

Mở terminal thứ hai:

```powershell
cd C:\Users\503\Downloads\IR-main
.\.venv\Scripts\Activate.ps1
py -3.11 register.py
```

Kết quả đúng thường có dạng:

```json
{
  "message": "Đăng ký thành công!",
  "student_id": "...",
  "server_url": "http://10.170.45.56:5000"
}
```

Nếu bị lỗi:

```text
Invalid URL 'http:/...': No host supplied
```

thì sửa `http:/` thành `http://`.

Nếu bị lỗi:

```text
404 Not Found
```

thì kiểm tra lại endpoint. Thường endpoint đúng là:

```text
/register
```

không phải:

```text
/competition/register
```

## 9. Bước 3: Chạy evaluate

Sau khi register thành công, vẫn giữ server đang chạy ở terminal thứ nhất.

Ở terminal thứ hai, chạy:

```powershell
py -3.11 evaluate.py
```

Nếu teacher server gọi được student server, quá trình đánh giá sẽ bắt đầu.

## 10. Thứ tự chạy nhanh

Mỗi lần chạy dự án, làm theo thứ tự này:

Terminal 1:

```powershell
cd C:\Users\503\Downloads\IR-main
.\.venv\Scripts\Activate.ps1
py -3.11 -m uvicorn server:app --host 0.0.0.0 --port 5000
```

Terminal 2:

```powershell
cd C:\Users\503\Downloads\IR-main
.\.venv\Scripts\Activate.ps1
py -3.11 register.py
py -3.11 evaluate.py
```

## 11. Một số lỗi thường gặp

### Lỗi 1: No module named fastapi

Nguyên nhân: cài thư viện vào Python khác, nhưng chạy bằng Python khác.

Cách sửa:

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 server.py
```

### Lỗi 2: No module named openai

Cài thiếu OpenAI SDK:

```powershell
py -3.11 -m pip install openai
```

### Lỗi 3: Path ./models/vietnamese-sbert not found

Chưa tải model local.

Cách sửa:

```powershell
py -3.11 download_model.py
```

### Lỗi 4: Teacher không gửi được document về student server

Kiểm tra các điểm sau:

- Student server đã bật chưa.
- Server chạy đúng port chưa, ví dụ `5000`.
- `STUDENT_SERVER_URL` trong `register.py` có đúng IP thật không.
- Không dùng IP VMware hoặc Loopback.
- Firewall Windows có chặn port không.

Mở port tạm thời cho Python nếu Windows hỏi quyền truy cập mạng.

### Lỗi 5: 404 Not Found khi register

Endpoint sai.

Sửa:

```python
url = f"{TEACHER_URL}/register"
```

Không dùng:

```python
url = f"{TEACHER_URL}/competition/register"
```

## 12. Ghi chú về cảnh báo LangChain

Nếu thấy cảnh báo:

```text
DeprecationWarning: langchain-community is being sunset
```

hoặc:

```text
LangChainDeprecationWarning: HuggingFaceEmbeddings was deprecated
```

thì đây chỉ là cảnh báo, không phải lỗi khiến server dừng.

Nếu muốn sửa về sau, có thể cài:

```powershell
py -3.11 -m pip install langchain-huggingface
```

và đổi import trong `server.py`:

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

thay cho:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
```
