import sys

MODELS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "keepitreal/vietnamese-sbert",
]


def main() -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    for name in MODELS:
        print(f"Downloading {name}...")
        SentenceTransformer(name)
        print(f"Done: {name}")


if __name__ == "__main__":
    main()
