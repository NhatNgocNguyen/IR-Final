"""Measure PhoBERT token/char ratio on sample Vietnamese legal text."""

from __future__ import annotations

from app.config import get_settings
from sentence_transformers import SentenceTransformer

SAMPLE_TEXTS = [
    (
        "ĐIỀU 15. Trách nhiệm của người điều khiển phương tiện giao thông\n"
        "1. Người điều khiển phương tiện tham gia giao thông đường bộ phải đi bên phải "
        "theo chiều đi của mình; với đường hai chiều phải đi đúng làn đường quy định.\n"
        "2. Khi chuyển hướng phải nhường đường cho người đi bộ, xe đi ngược chiều "
        "và các phương tiện khác theo quy định của pháp luật."
    ),
    (
        "CHƯƠNG III\n"
        "QUY ĐỊNH VỀ HỒ SƠ, THỦ TỤC CẤP PHÉP\n"
        "Điều 10. Hồ sơ đề nghị cấp giấy phép gồm: a) Đơn đề nghị theo mẫu quy định; "
        "b) Bản sao hợp lệ giấy chứng nhận đăng ký doanh nghiệp; c) Báo cáo kết quả "
        "thẩm định của cơ quan có thẩm quyền."
    ),
    (
        "Theo quy định tại khoản 2 Điều 5 Nghị định số 100/2019/NĐ-CP, tổ chức, cá nhân "
        "vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực đường bộ "
        "bị phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với hành vi không chấp hành "
        "hiệu lệnh của biển báo hiệu đường bộ."
    ),
]


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    settings = get_settings()
    model = SentenceTransformer(str(settings.embedding_model_path), local_files_only=True)
    tokenizer = model.tokenizer

    max_chunk_tokens = int(__import__("os").getenv("MAX_CHUNK_TOKENS", "250"))
    ratios: list[float] = []

    print(f"Model: {settings.embedding_model_path}")
    print(f"Target max_chunk_tokens: {max_chunk_tokens}")
    print()

    for index, sample in enumerate(SAMPLE_TEXTS, start=1):
        chars = len(sample)
        tokens = token_len(tokenizer, sample)
        ratio = tokens / chars if chars else 0.0
        ratios.append(ratio)
        print(f"Sample {index}: {chars} chars -> {tokens} tokens ({ratio:.3f} tok/char)")

    avg_ratio = sum(ratios) / len(ratios)
    suggested_chunk_size = int(max_chunk_tokens / avg_ratio) if avg_ratio > 0 else 500
    print()
    print(f"Average ratio: {avg_ratio:.3f} tokens/char")
    print(f"Suggested CHUNK_SIZE (chars) for ~{max_chunk_tokens} tokens: {suggested_chunk_size}")


if __name__ == "__main__":
    main()
