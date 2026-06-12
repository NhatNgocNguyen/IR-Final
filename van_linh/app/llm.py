from __future__ import annotations

import logging
import re

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import Settings


LOGGER = logging.getLogger(__name__)
ANSWER_PATTERN = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
OPTION_PATTERN = re.compile(
    r"(?ims)(?:^|\n)\s*([ABCD])[\.\):\-]\s*(.+?)(?=(?:\n\s*[ABCD][\.\):\-]\s)|\Z)"
)


class TeacherProxyTimeoutError(RuntimeError):
    """Teacher proxy timed out."""


class TeacherProxyRequestError(RuntimeError):
    """Teacher proxy request failed."""


class LlmService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.teacher_proxy_base_url,
            api_key=settings.student_id,
            timeout=settings.llm_timeout_seconds,
        )

    def answer_question(self, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(
            f"[Chunk {index + 1}]\n{chunk}" for index, chunk in enumerate(context_chunks)
        )
        prompt = self._build_prompt(question, context)

        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer Vietnamese legal multiple-choice questions using only the "
                            "provided legal text. Identify the best legal basis internally, then "
                            "return exactly one uppercase letter: A, B, C, or D."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
        except APITimeoutError as exc:
            LOGGER.exception("Teacher proxy request timed out")
            raise TeacherProxyTimeoutError("Teacher proxy request timed out") from exc
        except (APIConnectionError, APIStatusError) as exc:
            LOGGER.exception("Teacher proxy request failed")
            raise TeacherProxyRequestError("Teacher proxy request failed") from exc

        content = response.choices[0].message.content or ""
        answer = self._normalize_answer(content)
        if answer is None:
            raise RuntimeError(f"Could not parse answer from model output: {content!r}")
        return answer

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        options = LlmService._extract_options(question)
        if options:
            formatted_options = "\n".join(
                f"{label}. {content}" for label, content in options.items()
            )
            return (
                "Use the legal text below to answer the multiple-choice question.\n"
                "Prefer the option that is most directly supported by the legal text.\n"
                "If two options are similar, choose the one with the strongest explicit basis.\n"
                "Return only one letter: A, B, C, or D.\n\n"
                f"Legal text:\n{context}\n\n"
                f"Question:\n{LlmService._strip_options(question)}\n\n"
                f"Options:\n{formatted_options}"
            )

        return (
            "Use the legal text below to answer the Vietnamese multiple-choice question.\n"
            "Return only one letter: A, B, C, or D.\n\n"
            f"Legal text:\n{context}\n\n"
            f"Question:\n{question}"
        )

    @staticmethod
    def _extract_options(question: str) -> dict[str, str]:
        options = {
            label.upper(): " ".join(content.split())
            for label, content in OPTION_PATTERN.findall(question)
        }
        if {"A", "B", "C", "D"}.issubset(options):
            return options
        return {}

    @staticmethod
    def _strip_options(question: str) -> str:
        match = OPTION_PATTERN.search(question)
        if not match:
            return question.strip()
        return question[: match.start()].strip()

    @staticmethod
    def _normalize_answer(raw_text: str) -> str | None:
        text = raw_text.strip().upper()
        if text in {"A", "B", "C", "D"}:
            return text

        match = ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).upper()
        return None
