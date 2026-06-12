import re
import time
import os
import shutil

from fastapi import FastAPI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi

from openai import OpenAI

from config import STUDENT_HOST, STUDENT_PORT, TEACHER_PROXY_URL, STUDENT_ID
from schemas import UploadRequest, UploadResponse, AskRequest, AskResponse


app = FastAPI()

# =========================
# CONFIG RETRIEVAL
# =========================

VECTOR_DB_DIR = "./vector_db"
CHUNKS_FILE = "./vector_db/chunks.txt"

USE_BM25 = True

DENSE_TOP_K = 8
DENSE_FETCH_K = 20
BM25_TOP_K = 8
FINAL_TOP_K = 5

vector_db = None
bm25 = None
bm25_docs = []


# =========================
# EMBEDDING MODEL
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name="./models/vietnamese-sbert",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# =========================
# OPENAI CLIENT THROUGH TEACHER PROXY
# =========================

client = OpenAI(
    base_url=TEACHER_PROXY_URL,
    api_key=STUDENT_ID,
    timeout=25.0
)


# =========================
# UTILS
# =========================

def tokenize(text: str):
    """
    Tokenize đơn giản cho BM25.
    Dùng lowercase + bỏ dấu câu.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text.split()


def save_chunks_for_bm25(docs):
    """
    Lưu chunk ra file để khi tắt server bật lại vẫn load được BM25.
    """
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    texts = [doc.page_content for doc in docs]

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n-----CHUNK-----\n\n".join(texts))

    print(f"[BM25] chunks saved to {CHUNKS_FILE}")


def build_bm25_from_docs(docs):
    """
    Build BM25 từ list Document của LangChain.
    """
    global bm25, bm25_docs

    bm25_docs = [doc.page_content for doc in docs]
    tokenized_docs = [tokenize(text) for text in bm25_docs]

    bm25 = BM25Okapi(tokenized_docs)

    print(f"[BM25] BM25 created with {len(bm25_docs)} chunks")


def load_bm25_from_disk():
    """
    Load BM25 từ chunks.txt nếu đã có.
    """
    global bm25, bm25_docs

    if not os.path.exists(CHUNKS_FILE):
        print("[BM25] No chunks file found.")
        bm25 = None
        bm25_docs = []
        return

    try:
        print("[BM25] Loading chunks from disk...")

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        bm25_docs = [
            chunk.strip()
            for chunk in content.split("\n\n-----CHUNK-----\n\n")
            if chunk.strip()
        ]

        tokenized_docs = [tokenize(text) for text in bm25_docs]
        bm25 = BM25Okapi(tokenized_docs)

        print(f"[BM25] BM25 loaded with {len(bm25_docs)} chunks")

    except Exception as e:
        print(f"[BM25] Failed to load BM25: {e}")
        bm25 = None
        bm25_docs = []


def load_vector_db():
    """
    Load Chroma vector DB và BM25 nếu đã lưu trên disk.
    """
    global vector_db

    if os.path.exists(VECTOR_DB_DIR):
        try:
            print("[INIT] Loading existing Chroma vector DB...")

            vector_db = Chroma(
                persist_directory=VECTOR_DB_DIR,
                embedding_function=embedding_model
            )

            print("[INIT] Chroma vector DB loaded successfully.")

        except Exception as e:
            print(f"[INIT] Failed to load Chroma vector DB: {e}")
            vector_db = None
    else:
        print("[INIT] No existing vector DB found.")

    if USE_BM25:
        load_bm25_from_disk()


def build_options_text(req: AskRequest):
    """
    Lấy options từ note hoặc options.
    Teacher có thể gửi dạng:
    note = 'A. ...\\nB. ...\\nC. ...\\nD. ...'
    hoặc options = [...]
    """
    parts = []

    if getattr(req, "note", None):
        parts.append(req.note)

    if getattr(req, "options", None):
        parts.append("\n".join(req.options))

    return "\n".join(parts).strip()


def parse_options(options_text: str):
    """
    Parse options dạng:
    A. đáp án
    B. đáp án
    C. đáp án
    D. đáp án

    Hỗ trợ A. / A) / A:
    """
    if not options_text:
        return []

    pattern = r"([ABCD])[\.\):]\s*(.+?)(?=\n\s*[ABCD][\.\):]|\Z)"
    matches = re.findall(pattern, options_text, flags=re.S | re.I)

    options = []
    for label, text in matches:
        label = label.upper().strip()
        text = " ".join(text.strip().split())
        if label in ["A", "B", "C", "D"] and text:
            options.append((label, text))

    return options


def direct_option_match(context: str, options_text: str):
    """
    Nếu option xuất hiện trực tiếp trong context thì trả lời luôn.
    Cái này giúp các câu hỏi factual đơn giản chính xác hơn.
    """
    options = parse_options(options_text)

    if not options:
        return None

    context_lower = context.lower()

    candidates = []

    for label, option_text in options:
        normalized_option = option_text.lower().strip()

        # Bỏ qua option quá ngắn như "có", "không", "đúng", "sai"
        if len(normalized_option) < 3:
            continue

        if normalized_option in context_lower:
            candidates.append((label, len(normalized_option)))

    if not candidates:
        return None

    # Nếu nhiều option đều match, chọn option có text dài hơn để tránh match nhầm
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    return candidates[0][0]


def hybrid_retrieve(question: str):
    """
    Retrieval không dùng reranker:
    1. Dense retrieval bằng Chroma MMR
    2. Keyword retrieval bằng BM25
    3. Gộp và loại trùng
    4. Lấy FINAL_TOP_K chunk
    """
    global vector_db, bm25, bm25_docs

    candidates = []

    # 1. Dense retrieval bằng MMR
    if vector_db is not None:
        try:
            dense_docs = vector_db.max_marginal_relevance_search(
                question,
                k=DENSE_TOP_K,
                fetch_k=DENSE_FETCH_K,
                lambda_mult=0.7
            )

            for doc in dense_docs:
                candidates.append(doc.page_content)

            print(f"[RETRIEVE] dense MMR docs = {len(dense_docs)}")

        except Exception as e:
            print(f"[RETRIEVE] Dense MMR error = {e}")

            try:
                dense_docs = vector_db.similarity_search(question, k=DENSE_TOP_K)
                for doc in dense_docs:
                    candidates.append(doc.page_content)

                print(f"[RETRIEVE] dense fallback docs = {len(dense_docs)}")

            except Exception as e2:
                print(f"[RETRIEVE] Dense fallback error = {e2}")

    # 2. BM25 retrieval
    if USE_BM25 and bm25 is not None and bm25_docs:
        try:
            query_tokens = tokenize(question)
            scores = bm25.get_scores(query_tokens)

            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:BM25_TOP_K]

            for idx in top_indices:
                candidates.append(bm25_docs[idx])

            print(f"[RETRIEVE] BM25 docs = {len(top_indices)}")

        except Exception as e:
            print(f"[RETRIEVE] BM25 error = {e}")

    # 3. Deduplicate
    unique_texts = []
    seen = set()

    for text in candidates:
        key = text[:500]

        if key not in seen:
            seen.add(key)
            unique_texts.append(text)

    print(f"[RETRIEVE] unique candidates = {len(unique_texts)}")

    return unique_texts[:FINAL_TOP_K]


# =========================
# STARTUP
# =========================

@app.on_event("startup")
def startup_event():
    load_vector_db()


# =========================
# LOG REQUESTS
# =========================

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"\n[HTTP] {request.method} {request.url}")
    response = await call_next(request)
    print(f"[HTTP] status = {response.status_code}")
    return response


# =========================
# UPLOAD ENDPOINT
# =========================

@app.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest):
    global vector_db

    try:
        print("\n========== [UPLOAD RECEIVED] ==========")
        print(f"[UPLOAD] doc_id = {req.doc_id}")

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

        # Xóa vector DB cũ nếu có
        if os.path.exists(VECTOR_DB_DIR):
            print("[UPLOAD] Removing old vector DB...")
            shutil.rmtree(VECTOR_DB_DIR)

        os.makedirs(VECTOR_DB_DIR, exist_ok=True)

        # Lưu chunks cho BM25
        save_chunks_for_bm25(docs)

        # Build BM25
        if USE_BM25:
            build_bm25_from_docs(docs)

        # Build Chroma vector DB
        print("[UPLOAD] Creating Chroma vector DB...")

        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=VECTOR_DB_DIR
        )

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


# =========================
# ASK ENDPOINT
# =========================

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    global vector_db

    start_time = time.time()

    try:
        print("\n========== [ASK RECEIVED] ==========")
        print(f"[ASK] id = {getattr(req, 'id', None)}")
        print(f"[ASK] question = {req.question}")
        print(f"[ASK] note = {getattr(req, 'note', None)}")
        print(f"[ASK] options = {getattr(req, 'options', None)}")

        if vector_db is None:
            print("[ASK] vector_db is None, trying to load from disk...")
            load_vector_db()

        if vector_db is None:
            print("[ASK] vector_db still None")
            return AskResponse(answer="A", sources=[])

        # Lấy options từ note/options
        options_text = build_options_text(req)

        # Query nên bao gồm cả question và options để retrieval chính xác hơn
        retrieval_query = req.question

        if options_text:
            retrieval_query = req.question + "\n" + options_text

        retrieved_texts = hybrid_retrieve(retrieval_query)

        sources = [
            text[:300]
            for text in retrieved_texts
        ]

        context = "\n\n".join(retrieved_texts)

        # Direct match nếu option xuất hiện trực tiếp trong context
        direct_answer = direct_option_match(context, options_text)

        if direct_answer:
            print(f"[ASK] direct_answer = {direct_answer}")
            print(f"[ASK] elapsed = {time.time() - start_time:.2f}s")

            return AskResponse(
                answer=direct_answer,
                sources=sources
            )

        prompt = f"""
        Bạn là hệ thống trả lời câu hỏi trắc nghiệm dựa HOÀN TOÀN vào tài liệu được cung cấp.
        
        QUY TẮC:
        1. Chỉ sử dụng thông tin trong CONTEXT.
        2. Không suy đoán bằng kiến thức bên ngoài.
        3. Đọc kỹ QUESTION và OPTIONS.
        4. So khớp từng lựa chọn A/B/C/D với CONTEXT.
        5. Nếu đáp án đúng xuất hiện trực tiếp trong CONTEXT thì chọn lựa chọn tương ứng.
        6. Chỉ trả về duy nhất 1 ký tự: A hoặc B hoặc C hoặc D.
        7. Không giải thích.
        
        ================ CONTEXT ================
        {context}
        
        ================ QUESTION ================
        {req.question}
        
        ================ OPTIONS ================
        {options_text}
        
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


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=STUDENT_HOST,
        port=STUDENT_PORT,
        reload=False
    )
