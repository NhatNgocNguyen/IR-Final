"""
download_model.py — Download embedding models for IR-Final student servers.

Run once while you have internet access, before the offline exam.

Models required:
  nhat  →  paraphrase-multilingual-MiniLM-L12-v2  (~420 MB, Hugging Face cache)
  bell  →  keepitreal/vietnamese-sbert            (saved to bell/models/vietnamese-sbert)

LLM (gpt-4o-mini) is NOT downloaded here — both servers call the Teacher proxy at exam time.

Usage:
  python download_model.py              # download both
  python download_model.py --target nhat
  python download_model.py --target bell
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS = {
    "nhat": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "save_dir": None,  # uses Hugging Face cache; nhat/embedder.py loads by name
        "project": "nhat",
        "size_hint": "~420 MB",
        "server_entry": "python nhat/main.py",
    },
    "bell": {
        "name": "keepitreal/vietnamese-sbert",
        "save_dir": ROOT / "bell" / "models" / "vietnamese-sbert",
        "project": "bell",
        "size_hint": "~400 MB",
        "server_entry": "python bell/server.py",
    },
}


def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌  sentence-transformers is not installed.")
        print("    Run from IR-Final:  pip install sentence-transformers")
        sys.exit(1)
    return SentenceTransformer


def download_nhat() -> None:
    cfg = MODELS["nhat"]
    print(f"\n{'=' * 60}")
    print(f"  NHAT — {cfg['name']}")
    print(f"  Size: {cfg['size_hint']}  |  Storage: Hugging Face cache")
    print(f"{'=' * 60}\n")

    SentenceTransformer = _load_sentence_transformer()
    start = time.time()

    try:
        model = SentenceTransformer(cfg["name"])
    except Exception as exc:
        print(f"❌  Download failed: {exc}")
        sys.exit(1)

    elapsed = time.time() - start

    test_sentences = [
        "RAG là gì trong xử lý ngôn ngữ tự nhiên?",
        "Retrieval-Augmented Generation combines retrieval with generation.",
    ]
    vecs = model.encode(test_sentences, normalize_embeddings=True)
    sim = float(vecs[0] @ vecs[1])

    print(f"✅  Downloaded in {elapsed:.1f}s")
    print(f"   Embedding dim : {vecs.shape[1]}")
    print(f"   Vi↔En similarity smoke test: {sim:.3f}")
    print(f"\n   Start server: {cfg['server_entry']}")


def download_bell() -> None:
    cfg = MODELS["bell"]
    save_dir: Path = cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  BELL — {cfg['name']}")
    print(f"  Size: {cfg['size_hint']}  |  Save to: {save_dir}")
    print(f"{'=' * 60}\n")

    SentenceTransformer = _load_sentence_transformer()
    start = time.time()

    try:
        model = SentenceTransformer(cfg["name"])
        model.save(str(save_dir))
    except Exception as exc:
        print(f"❌  Download failed: {exc}")
        sys.exit(1)

    elapsed = time.time() - start

    test_sentences = [
        "Truy xuất thông tin là gì?",
        "Information retrieval là một lĩnh vực trong khoa học máy tính.",
    ]
    vecs = model.encode(test_sentences, normalize_embeddings=True)
    sim = float(vecs[0] @ vecs[1])

    print(f"✅  Downloaded in {elapsed:.1f}s")
    print(f"   Saved to      : {save_dir.resolve()}")
    print(f"   Embedding dim : {vecs.shape[1]}")
    print(f"   Vi similarity smoke test: {sim:.3f}")
    print(f"\n   bell/server.py uses: model_name=\"./models/vietnamese-sbert\"")
    print(f"   Start server: cd bell && python server.py")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download embedding models for nhat and bell student servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--target",
        choices=["nhat", "bell", "all"],
        default="all",
        help="Which project model to download (default: all)",
    )
    args = parser.parse_args()

    print("IR-Final model downloader")
    print("Requires internet. LLM (gpt-4o-mini) is served by Teacher proxy — not downloaded here.\n")

    if args.target in ("nhat", "all"):
        download_nhat()

    if args.target in ("bell", "all"):
        download_bell()

    print("\n✅  Done. You can run the servers offline (Teacher LLM proxy still needs LAN at exam time).")


if __name__ == "__main__":
    main()
