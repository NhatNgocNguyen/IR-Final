"""
download_model.py — Run this ONCE while you have internet access.

Downloads the multilingual sentence-transformer model to the local
Hugging Face cache so the exam server works fully offline.

Usage:
  python download_model.py
"""

import sys
import time

from config import EMBEDDING_MODEL


def main() -> None:
    print(f"Downloading embedding model: {EMBEDDING_MODEL}")
    print("(This requires internet access — ~420 MB download)")
    print()

    start = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
    except ImportError:
        print("❌  sentence-transformers is not installed.")
        print("    Run:  pip install sentence-transformers")
        sys.exit(1)
    except Exception as exc:
        print(f"❌  Download failed: {exc}")
        sys.exit(1)

    elapsed = time.time() - start

    # Quick smoke test
    test_sentences = [
        "RAG là gì trong xử lý ngôn ngữ tự nhiên?",
        "Retrieval-Augmented Generation combines retrieval with generation.",
    ]
    vecs = model.encode(test_sentences, normalize_embeddings=True)
    sim  = float(vecs[0] @ vecs[1])

    print(f"✅  Model downloaded in {elapsed:.1f}s")
    print(f"   Embedding dim : {vecs.shape[1]}")
    print(f"   Cross-lingual cosine similarity (vi↔en): {sim:.3f}")
    print()
    print("You can now run the server offline:")
    print("  python main.py")


if __name__ == "__main__":
    main()
