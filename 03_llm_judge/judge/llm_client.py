"""
OpenRouter LLM client for judge calls.

Uses the OpenAI SDK pointed at OpenRouter, matching the pattern in the
student's existing code (own_ragas_metrics_in-use.py).

Structured JSON output is requested via response_format with json_schema,
so every call returns a guaranteed JSON object.
"""

import json
import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file if present (searches current dir and parent dirs)
load_dotenv()

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Set it in your .env file or shell environment."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_judge_llm(
    prompt: str,
    response_schema: dict,
    schema_name: str,
    model: str,
    provider: str | None = None,
    max_tokens: int = 4096,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    temperature: float = 0.0,
) -> dict:
    """
    Call the judge LLM and return the parsed JSON response.

    Parameters
    ----------
    prompt : str
        The full prompt text (system + user merged, or just user).
    response_schema : dict
        JSON Schema dict for the expected response object.
    schema_name : str
        Name for the json_schema response_format entry.
    model : str
        OpenRouter model identifier, e.g. "google/gemini-2.5-pro".
    provider : str | None
        Optional provider pin (e.g. "Google"). If None, OpenRouter routes freely.
    max_tokens : int
        Maximum tokens for the response.
    max_retries : int
        Number of retry attempts on transient failures.
    retry_delay : float
        Seconds to wait between retries.

    Returns
    -------
    dict
        Parsed JSON response matching response_schema.

    Raises
    ------
    RuntimeError
        If all retries are exhausted.
    """
    client = get_client()

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": response_schema,
        },
    }

    request_params: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": response_format,
    }

    # Optionally pin to a specific provider
    if provider is not None:
        request_params["extra_body"] = {
            "provider": {
                "order": [provider],
                "allow_fallbacks": False,
            }
        }

    last_exception: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(
                "Judge LLM call attempt %d/%d | model=%s", attempt, max_retries, model
            )
            response = client.chat.completions.create(**request_params)
            content = response.choices[0].message.content

            # Log token usage
            if response.usage:
                logger.debug(
                    "Token usage: prompt=%d, completion=%d, total=%d",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )

            parsed = json.loads(content)
            return parsed

        except json.JSONDecodeError as e:
            logger.error(
                "JSON decode error on attempt %d: %s. Response content: %s",
                attempt,
                str(e),
                content if "content" in dir() else "<no content>",
            )
            last_exception = e
        except Exception as e:
            logger.warning("LLM call error on attempt %d: %s", attempt, str(e))
            last_exception = e

        if attempt < max_retries:
            logger.info("Retrying in %.1f seconds...", retry_delay)
            time.sleep(retry_delay)

    raise RuntimeError(
        f"All {max_retries} attempts to call judge LLM failed. "
        f"Last error: {last_exception}"
    )
