# IR-Final — Setup Guide

Hướng dẫn cài dependencies và tải embedding model cho các student server trong repo này.

---

## 1. Yêu cầu

- Python **3.10+** (repo dùng `.python-version` → 3.14)
- Internet (chỉ cho bước `pip install` và `download_model.py`)
- ~1 GB dung lượng trống (venv + models)

---

## 2. Cài dependencies

Từ thư mục gốc `IR-Final/`, cài mọi `requirements.txt` trong repo:

```bash
find . -name "requirements.txt" -exec uv pip install -r {} \;
```

Hoặc dùng `pip`:

```bash
find . -name "requirements.txt" -exec pip install -r {} \;
```

---

## 3. Tải embedding models

Chạy **một lần** khi còn internet:

```bash
python download_model.py
```

Thêm model mới trong `download_model.py` → list `MODELS`.
