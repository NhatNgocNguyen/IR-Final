# IR-Final — Setup Guide

Hướng dẫn cài dependencies và tải embedding model cho các student server trong repo này.

---

## 1. Yêu cầu

- Python **3.10+** (repo dùng `.python-version` → 3.14)
- Internet (chỉ cho bước `pip install` và `download_model.py`)
- ~1 GB dung lượng trống (venv + models)

---

## 2. Cài dependencies

Một file `requirements.txt` gộp đủ cho cả **nhat**, **bell**, và **van_linh**:

```bash
pip install -r requirements.txt
```

Bao gồm: FastAPI, uvicorn, OpenAI SDK, sentence-transformers, numpy, rank-bm25, requests, httpx, python-dotenv, langchain, chromadb.

---

## 3. Tải embedding models

Chạy **một lần** khi còn internet, từ thư mục `IR-Final/`:

```bash
python download_model.py
```

Tải cả hai model:

| Target | Model | Lưu ở đâu |
|---|---|---|
| `nhat` | `paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face cache (~420 MB) |
| `bell` | `keepitreal/vietnamese-sbert` | `bell/models/vietnamese-sbert/` |

Chỉ tải một project:

```bash
python download_model.py --target nhat
python download_model.py --target bell
```

Script sẽ chạy smoke test sau khi tải xong. Nếu thấy `✅ Downloaded`, có thể chạy server offline (embedding). Teacher LLM proxy vẫn cần LAN lúc thi.
