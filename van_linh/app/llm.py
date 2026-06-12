from __future__ import annotations

import logging
import math
import re
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import Settings


LOGGER = logging.getLogger(__name__)
DEFAULT_PROMPT_VARIANT = 2
LLM_MAX_OUTPUT_TOKENS = 8

ANSWER_PATTERNS = [
    re.compile(r"^([ABCD])$", re.IGNORECASE),
    re.compile(
        r"(?:ĐÁP\s*ÁN|DAP\s*AN|ANSWER|ANS|IS)\s*[:\-]?\s*([ABCD])\b",
        re.IGNORECASE,
    ),
    re.compile(r"[(\[]\s*([ABCD])\s*[)\]]", re.IGNORECASE),
    re.compile(r"\b([ABCD])[.)]", re.IGNORECASE),
    re.compile(r"(?:LÀ|LA)\s+([ABCD])\b", re.IGNORECASE),
    re.compile(r"\b([ABCD])\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"\b([ABCD])\b", re.IGNORECASE),
]

PROMPT_TEMPLATES: dict[int, dict[str, str]] = {
    1: {
        "system": (
            "Bạn là hệ thống trả lời câu hỏi trắc nghiệm tiếng Việt.\n"
            "Nhiệm vụ: chọn đúng một đáp án A, B, C hoặc D dựa trên tài liệu được cung cấp.\n"
            "Chỉ trả về đúng một ký tự in hoa: A, B, C hoặc D. Không giải thích."
        ),
        "user": (
            "Trả lời câu hỏi trắc nghiệm tiếng Việt dưới đây, dựa trên nội dung tài liệu.\n"
            "\n"
            "Quy tắc:\n"
            "1. Xác định câu hỏi đang hỏi đáp án ĐÚNG, đáp án SAI, hay ngoại lệ.\n"
            "2. Đọc kỹ từng phương án A, B, C, D và so sánh nghĩa với tài liệu — không chỉ khớp từ ngữ.\n"
            "3. Với câu định nghĩa, liệt kê, số lượng, điều kiện: ưu tiên bằng chứng được nêu rõ trong tài liệu.\n"
            '4. Với câu hỏi "câu nào SAI / nhận định SAI / khẳng định SAI": loại từng phương án, '
            "chọn phương án không khớp tài liệu.\n"
            "5. Với câu điền khuyết (.....): ghép từng phương án vào chỗ trống, "
            "chọn phương án tạo câu đúng nghĩa nhất.\n"
            "6. Nếu tài liệu không đủ rõ, chọn phương án được hỗ trợ tốt nhất bởi ngữ cảnh.\n"
            "7. Trả về đúng một ký tự: A, B, C hoặc D.\n"
            "\n"
            "Tài liệu tham khảo:\n"
            "{context}\n"
            "\n"
            "Câu hỏi và các phương án:\n"
            "{question}\n"
            "\n"
            "Đáp án:"
        ),
    },
    2: {
        "system": (
            "Bạn là hệ thống trả lời câu hỏi trắc nghiệm.\n"
            "Nhiệm vụ: chọn một đáp án A, B, C hoặc D tốt nhất, dùng tài liệu làm nguồn chính "
            "và kiến thức nền để hiểu, suy luận khi tài liệu chưa nói thẳng.\n"
            "\n"
            "Cách suy luận:\n"
            "1. Đọc hết các đoạn tài liệu và toàn bộ câu hỏi kèm 4 phương án.\n"
            "2. Ưu tiên phương án được tài liệu hỗ trợ trực tiếp.\n"
            "3. Dùng kiến thức nền để giải thích thuật ngữ, lấp chỗ trống, suy luận khi tài liệu mơ hồ hoặc im lặng.\n"
            "4. So sánh từng phương án; loại các phương án sai rõ ràng.\n"
            "5. Nếu tài liệu và suy luận mâu thuẫn, chọn phương án đáng tin hơn cho câu hỏi cụ thể này.\n"
            "6. Chỉ trả về một chữ cái in hoa: A, B, C hoặc D. Không giải thích."
        ),
        "user": (
            "Trả lời câu hỏi trắc nghiệm dưới đây.\n"
            "Dùng các đoạn tài liệu làm bằng chứng chính; bổ sung kiến thức nền khi cần để phân biệt các phương án.\n"
            "\n"
            "Các bước:\n"
            "- Đọc từng đoạn [Chunk 1], [Chunk 2], ...\n"
            "- Xác định câu hỏi đang hỏi gì và mỗi phương án khẳng định điều gì.\n"
            "- Kết hợp bằng chứng tài liệu với suy luận để chọn phương án đúng nhất.\n"
            "\n"
            "=== TÀI LIỆU ===\n"
            "{context}\n"
            "=== HẾT TÀI LIỆU ===\n"
            "\n"
            "=== CÂU HỎI ===\n"
            "{question}\n"
            "=== HẾT CÂU HỎI ===\n"
            "\n"
            "Trả lời bằng đúng một chữ cái: A, B, C hoặc D."
        ),
    },
    3: {
        "system": "Bạn trả lời câu hỏi trắc nghiệm tiếng Việt. Chỉ trả về A, B, C hoặc D.",
        "user": (
            "Dựa trên tài liệu, trả lời câu hỏi trắc nghiệm.\n"
            "\n"
            "Bước 1 — Đọc loại câu hỏi:\n"
            '- "đúng", "chính xác", "phù hợp" → chọn phương án ĐÚNG theo tài liệu.\n'
            '- "sai", "không đúng", "không phù hợp" → chọn phương án SAI (không khớp tài liệu).\n'
            '- "ngoại lệ", "trừ", "không bao gồm" → chọn phương án là ngoại lệ.\n'
            "\n"
            "Bước 2 — Với mỗi phương án A–D:\n"
            "- Phương án nói gì?\n"
            "- Tài liệu có ủng hộ hay bác bỏ?\n"
            "\n"
            "Bước 3 — Chọn một đáp án duy nhất.\n"
            "\n"
            "Tài liệu:\n"
            "{context}\n"
            "\n"
            "Câu hỏi:\n"
            "{question}\n"
            "\n"
            "Đáp án (một chữ cái):"
        ),
    },
    4: {
        "system": "Chỉ trả về đúng một ký tự: A, B, C hoặc D.",
        "user": (
            "Trả lời câu hỏi trắc nghiệm tiếng Việt bằng nội dung tài liệu dưới đây.\n"
            "\n"
            "Quy tắc:\n"
            "1. Xác định câu hỏi hỏi đáp án đúng, sai hay ngoại lệ.\n"
            "2. So sánh nghĩa từng phương án với tài liệu.\n"
            "3. Với định nghĩa, liệt kê, bước, số lượng: ưu tiên bằng chứng rõ ràng.\n"
            "4. Chọn phương án có căn cứ trực tiếp mạnh nhất.\n"
            "5. Nếu thiếu bằng chứng, chọn phương án gần nhất với tài liệu.\n"
            "6. Chỉ trả về A, B, C hoặc D.\n"
            "\n"
            "Tài liệu:\n"
            "{context}\n"
            "\n"
            "Câu hỏi:\n"
            "{question}\n"
            "\n"
            "Đáp án:"
        ),
    },
}


class TeacherProxyTimeoutError(RuntimeError):
    """Teacher proxy timed out."""


class TeacherProxyRequestError(RuntimeError):
    """Teacher proxy request failed."""


def resolve_prompt_variant(variant: int) -> int:
    if variant in PROMPT_TEMPLATES:
        return variant
    return DEFAULT_PROMPT_VARIANT


def estimate_gpt_tokens(text: str, chars_per_token: float) -> int:
    if not text or chars_per_token <= 0:
        return 0
    return math.ceil(len(text) / chars_per_token)


def build_prompt(question: str, context: str, variant: int) -> tuple[str, str]:
    resolved_variant = resolve_prompt_variant(variant)
    template = PROMPT_TEMPLATES[resolved_variant]
    user_prompt = template["user"].format(context=context, question=question)
    return template["system"], user_prompt


def estimate_prompt_tokens(
    question: str,
    context: str,
    variant: int,
    chars_per_token: float,
) -> int:
    system_prompt, user_prompt = build_prompt(question, context, variant)
    return estimate_gpt_tokens(system_prompt, chars_per_token) + estimate_gpt_tokens(
        user_prompt, chars_per_token
    )


def _prompt_overhead_tokens(
    question: str,
    variant: int,
    chars_per_token: float,
) -> int:
    resolved_variant = resolve_prompt_variant(variant)
    template = PROMPT_TEMPLATES[resolved_variant]
    system_prompt = template["system"]
    user_without_context = template["user"].format(context="", question=question)
    return estimate_gpt_tokens(system_prompt, chars_per_token) + estimate_gpt_tokens(
        user_without_context, chars_per_token
    )


def _join_context_chunks(chunks: list[str]) -> str:
    return "\n\n".join(f"[Chunk {index + 1}]\n{chunk}" for index, chunk in enumerate(chunks))


def fit_context_to_token_budget(
    context_chunks: list[str],
    question: str,
    variant: int,
    *,
    max_input_tokens: int,
    chars_per_token: float,
) -> tuple[str, int]:
    if not context_chunks:
        return "", 0

    overhead_tokens = _prompt_overhead_tokens(question, variant, chars_per_token)
    context_budget_tokens = max(0, max_input_tokens - overhead_tokens)

    selected_chunks: list[str] = []
    for chunk in context_chunks:
        candidate_chunks = selected_chunks + [chunk]
        candidate_context = _join_context_chunks(candidate_chunks)
        if estimate_gpt_tokens(candidate_context, chars_per_token) <= context_budget_tokens:
            selected_chunks.append(chunk)
            continue

        if selected_chunks:
            break

        max_chars = max(1, int(context_budget_tokens * chars_per_token))
        truncated = chunk[:max_chars]
        if len(truncated) < len(chunk):
            truncated = truncated.rsplit("\n", 1)[0].strip() or truncated
        selected_chunks.append(truncated)
        break

    return _join_context_chunks(selected_chunks), len(selected_chunks)


def normalize_answer(raw_text: str) -> str | None:
    text = raw_text.strip()
    if not text:
        return None

    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()

    return None


class LlmService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.teacher_proxy_base_url,
            api_key=settings.student_id,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    def answer_question(self, question: str, context_chunks: list[str]) -> str:
        started_at = time.perf_counter()
        max_input_tokens = max(
            1,
            self._settings.max_prompt_tokens - self._settings.prompt_token_margin,
        )
        context, used_chunk_count = fit_context_to_token_budget(
            context_chunks,
            question,
            self._settings.prompt_variant,
            max_input_tokens=max_input_tokens,
            chars_per_token=self._settings.gpt_chars_per_token,
        )

        system_prompt, user_prompt = build_prompt(
            question,
            context,
            self._settings.prompt_variant,
        )
        estimated_tokens = estimate_gpt_tokens(system_prompt, self._settings.gpt_chars_per_token)
        estimated_tokens += estimate_gpt_tokens(user_prompt, self._settings.gpt_chars_per_token)
        estimated_tokens += LLM_MAX_OUTPUT_TOKENS

        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
            )
        except APITimeoutError as exc:
            LOGGER.exception("Teacher proxy request timed out")
            raise TeacherProxyTimeoutError("Teacher proxy request timed out") from exc
        except (APIConnectionError, APIStatusError) as exc:
            LOGGER.exception("Teacher proxy request failed")
            raise TeacherProxyRequestError("Teacher proxy request failed") from exc

        content = response.choices[0].message.content or ""
        answer = normalize_answer(content)
        if answer is None:
            raise RuntimeError(f"Could not parse answer from model output: {content!r}")
        LOGGER.info(
            "Teacher proxy returned answer=%s (prompt_variant=%s, ~%s tokens, "
            "chunks=%s/%s, budget=%s) in %.3fs",
            answer,
            resolve_prompt_variant(self._settings.prompt_variant),
            estimated_tokens,
            used_chunk_count,
            len(context_chunks),
            self._settings.max_prompt_tokens,
            time.perf_counter() - started_at,
        )
        return answer
