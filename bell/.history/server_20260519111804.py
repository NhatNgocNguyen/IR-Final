import re
import time

from fastapi import FastAPI

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from openai import OpenAI

from config import STUDENT_HOST, STUDENT_PORT, TEACHER_PROXY_URL, STUDENT_ID
from schemas import UploadRequest, UploadResponse, AskRequest, AskResponse


app = FastAPI()

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
            chunk_size=400,
            chunk_overlap=80
        )

        docs = splitter.create_documents([req.text])

        print(f"[UPLOAD] total chunks = {len(docs)}")

        # xem thử từng chunk
        for i, doc in enumerate(docs[:5]):
            print(f"\n----- CHUNK {i} -----")
            print(doc.page_content[:500])

        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )

        print("[UPLOAD] vector db created")

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
        print("\n========== [ASK RECEIVED] ==========")
        print(f"[ASK] question = {req.question}")

        if vector_db is None:
            print("[ASK] vector_db is None")
            return AskResponse(answer="A", sources=[])

        retrieved_docs = vector_db.similarity_search(req.question, k=3)

        sources = [
            doc.page_content[:300]
            for doc in retrieved_docs
        ]

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

        prompt = f"""
        Chỉ trả lời đúng 1 ký tự A hoặc B hoặc C hoặc D.
        Không giải thích.

        Tài liệu:
        {context}

        Câu hỏi:
        {req.question}

        Đáp án:
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