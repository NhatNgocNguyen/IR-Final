# download_model.py
# Chạy file này để tải model keepitreal/vietnamese-sbert về local:
#   py -3.11 download_model.py

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

    print("\nDone!")
    print("Model đã được tải về local.")
    print(f"Đường dẫn model: {SAVE_DIR.resolve()}")
    print("\nBây giờ trong server.py bạn có thể dùng:")
    print('model_name="./models/vietnamese-sbert"')

if __name__ == "__main__":
    main()
