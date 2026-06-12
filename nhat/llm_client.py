"""
llm_client.py — Wrapper around the teacher's LLM proxy.

Uses the openai SDK with a custom base_url pointing to the proxy.
Includes exponential-backoff retry for transient network errors.
"""

import logging
import time
from typing import List, Dict, Any

from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

from config import (
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    PROXY_BASE_URL,
    STUDENT_ID,
)

log = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=PROXY_BASE_URL,
            api_key=STUDENT_ID,
            timeout=LLM_TIMEOUT,
            max_retries=0,   # we handle retries ourselves
        )
    return _client


def chat(
    messages: List[Dict[str, str]],
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    Send a chat request to the proxy LLM.

    Returns the assistant message content as a string.
    Raises on permanent failure after LLM_MAX_RETRIES attempts.
    """
    client   = _get_client()
    last_exc = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()

        except RateLimitError as exc:
            wait = 2 ** (attempt + 1)
            log.warning("Rate-limited (attempt %d/%d), sleeping %ds: %s",
                        attempt + 1, LLM_MAX_RETRIES, wait, exc)
            time.sleep(wait)
            last_exc = exc

        except APIConnectionError as exc:
            wait = 1 * (attempt + 1)
            log.warning("Connection error (attempt %d/%d), sleeping %ds: %s",
                        attempt + 1, LLM_MAX_RETRIES, wait, exc)
            time.sleep(wait)
            last_exc = exc

        except APIStatusError as exc:
            # 5xx → retry; 4xx → don't retry
            if exc.status_code >= 500:
                wait = 2 ** attempt
                log.warning("Server error %d (attempt %d/%d), sleeping %ds",
                            exc.status_code, attempt + 1, LLM_MAX_RETRIES, wait)
                time.sleep(wait)
                last_exc = exc
            else:
                log.error("Client error %d: %s", exc.status_code, exc)
                raise

        except Exception as exc:
            wait = 2 ** attempt
            log.warning("Unexpected error (attempt %d/%d), sleeping %ds: %s",
                        attempt + 1, LLM_MAX_RETRIES, wait, exc)
            time.sleep(wait)
            last_exc = exc

    log.error("LLM call failed after %d attempts", LLM_MAX_RETRIES)
    raise RuntimeError(f"LLM unavailable after {LLM_MAX_RETRIES} retries") from last_exc
