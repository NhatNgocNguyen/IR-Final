"""
config.py — All configuration in one place.
Edit STUDENT_ID and MY_SERVER_URL before the exam.
"""
import os

# ================================================================
# THAY ĐỔI 2 DÒNG NÀY TRƯỚC KHI THI
# ================================================================
STUDENT_ID    = os.getenv("STUDENT_ID",    "YOUR_MSSV_HERE")
MY_SERVER_URL = os.getenv("MY_SERVER_URL", "http://192.168.1.X:5000")

# ================================================================
# SERVER
# ================================================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = int(os.getenv("PORT", "5000"))

# ================================================================
# TEACHER / PROXY ENDPOINTS
# ================================================================
TEACHER_BASE_URL = "http://192.168.50.218:8000/api/v1"
PROXY_BASE_URL   = f"{TEACHER_BASE_URL}/proxy"

# ================================================================
# LLM
# ================================================================
LLM_MODEL       = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0          # deterministic → reproducible MCQ answers
LLM_MAX_TOKENS  = 8            # only need 1 letter
LLM_MAX_RETRIES = 3
LLM_TIMEOUT     = 50           # seconds — teacher waits max 60s per /ask

# ================================================================
# PROMPT / CONTEXT BUDGET
# ================================================================
# Teacher says max prompt = 4000 chars.
# Budget breakdown (chars):
#   system prompt      ~250
#   question           ~300
#   preamble labels    ~150
#   context chunks     ← remainder
MAX_TOTAL_PROMPT_CHARS = 4000
SYSTEM_PROMPT_OVERHEAD = 250
QUESTION_OVERHEAD      = 300
PREAMBLE_OVERHEAD      = 150
MAX_CONTEXT_CHARS      = (MAX_TOTAL_PROMPT_CHARS
                          - SYSTEM_PROMPT_OVERHEAD
                          - QUESTION_OVERHEAD
                          - PREAMBLE_OVERHEAD)   # ≈ 3 100

# ================================================================
# CHUNKING
# ================================================================
CHUNK_SIZE    = 450   # chars per chunk — fits ~6–7 chunks in context
CHUNK_OVERLAP = 90    # overlap for continuity
MIN_CHUNK_LEN = 40    # discard tiny fragments

# ================================================================
# RETRIEVAL
# ================================================================
TOP_K_RETRIEVE = 8    # candidates from hybrid search
TOP_K_FINAL    = 6    # chunks sent to LLM

# Hybrid weights (BM25 handles exact Vietnamese terms well)
BM25_WEIGHT  = 0.40
DENSE_WEIGHT = 0.60

# ================================================================
# EMBEDDING MODEL
# ================================================================
# keepitreal/vietnamese-sbert — PhoBERT-based, tuned for Vietnamese
EMBEDDING_MODEL      = "keepitreal/vietnamese-sbert"
EMBEDDING_BATCH_SIZE = 64

# ================================================================
# PERSISTENCE
# ================================================================
DATA_DIR          = os.path.join(os.path.dirname(__file__), "data")
VECTORDB_PATH     = os.path.join(DATA_DIR, "vectordb.pkl")
