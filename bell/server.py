import re
import time
import os
import shutil

from fastapi import FastAPI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from openai import OpenAI

from config import STUDENT_HOST, STUDENT_PORT, TEACHER_PROXY_URL, STUDENT_ID
from schemas import UploadRequest, UploadResponse, AskRequest, AskResponse


app = FastAPI()

# Thư mục lưu vector database
VECTOR_DB_DIR = "./vector_db"

vector_db = None


embedding_model = HuggingFaceEmbeddings(
    model_name="./models/vietnamese-sbert",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


client = OpenAI(
    base_url=TEACHER_PROXY_URL,
    api_key=STUDENT_ID,
    timeout=25.0
)


def load_vector_db():
    """
    Load vector DB đã lưu từ disk nếu tồn tại.
    Dùng khi server vừa bật lên.
    """
    global vector_db

    if os.path.exists(VECTOR_DB_DIR):
        try:
            print("[INIT] Loading existing vector DB...")
            vector_db = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=embedding_model
            )
            print("[INIT] Vector DB loaded successfully.")
        except Exception as e:
            print(f"[INIT] Failed to load vector DB: {e}")
            vector_db = None
    else:
        print("[INIT] No existing vector DB found.")


@app.on_event("startup")
def startup_event():
    load_vector_db()


@app.middleware("http")
async def log_requests(request, call_next):
    print(f"\n[HTTP] {request.method} {request.url}")
    response = await call_next(request)
    print(f"[HTTP] status = {response.status_code}")
    return response


@app.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest):
    global vector_db

    try:
        print("\n========== [UPLOAD RECEIVED] ==========")
        print(f"[UPLOAD] doc_id = {req.doc_id}")

        print("\n========== FULL DOCUMENT ==========\n")
        print(req.text)
        print("\n========== END DOCUMENT ==========\n")

        print(f"[UPLOAD] text length = {len(req.text)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        docs = splitter.create_documents([req.text])

        print(f"[UPLOAD] total chunks = {len(docs)}")

        for i, doc in enumerate(docs[:5]):
            print(f"\n----- CHUNK {i} -----")
            print(doc.page_content[:500])

        # Nếu đã có vector DB cũ thì xóa để tạo lại từ document mới
        if os.path.exists(VECTOR_DB_DIR):
            print("[UPLOAD] Removing old vector DB...")
            shutil.rmtree(VECTOR_DB_DIR)

        print("[UPLOAD] Creating vector DB...")

        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=VECTOR_DB_DIR
        )

        # Một số bản Chroma tự persist, nhưng gọi thêm để chắc chắn
        try:
            vector_db.persist()
        except Exception:
            pass

        print("[UPLOAD] vector DB created and saved")
        print(f"[UPLOAD] saved at: {os.path.abspath(VECTOR_DB_DIR)}")

        return UploadResponse(
            status="success",
            doc_id=req.doc_id,
            chunks=len(docs)
        )

    except Exception as e:
        print(f"[UPLOAD] ERROR = {e}")

        return UploadResponse(
            status="error",
            doc_id=req.doc_id,
            chunks=0
        )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    global vector_db

    start_time = time.time()

    try:
        print(f"[ASK] question = {req.question}")
        print(f"[ASK] note = {req.note}")
        print(f"[ASK] options = {req.options}")

        # Nếu server mới bật lại mà vector_db chưa nằm trong RAM,
        # thử load lại từ thư mục vector_db
        if vector_db is None:
            print("[ASK] vector_db is None, trying to load from disk...")
            load_vector_db()

        if vector_db is None:
            print("[ASK] vector_db still None")
            return AskResponse(answer="A", sources=[])

        retrieved_docs = vector_db.similarity_search(req.question, k=5)

        sources = [
            doc.page_content[:300]
            for doc in retrieved_docs
        ]

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        prompt = f"""
Bạn là hệ thống trả lời câu hỏi trắc nghiệm dựa HOÀN TOÀN vào tài liệu được cung cấp.

QUY TẮC:
1. Chỉ sử dụng thông tin trong CONTEXT.
2. Không suy đoán bằng kiến thức bên ngoài.
3. Nếu thấy đáp án xuất hiện trực tiếp trong CONTEXT thì chọn đúng nguyên văn.
4. Chỉ trả về duy nhất 1 ký tự: A hoặc B hoặc C hoặc D.
5. Không giải thích.

================ CONTEXT ================
{context}

================ QUESTION ================
{req.question}

================ ANSWER ================
"""

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5
        )

        raw_answer = res.choices[0].message.content.strip().upper()
        match = re.search(r"[ABCD]", raw_answer)
        answer = match.group(0) if match else "A"

        print(f"[ASK] raw_answer = {raw_answer}")
        print(f"[ASK] final_answer = {answer}")
        print(f"[ASK] elapsed = {time.time() - start_time:.2f}s")

        return AskResponse(
            answer=answer,
            sources=sources
        )

    except Exception as e:
        print(f"[ASK] ERROR = {e}")
        return AskResponse(answer="A", sources=[])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=STUDENT_HOST,
        port=STUDENT_PORT,
        reload=False
    )
