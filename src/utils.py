import time

try:
    from langgraph.errors import GraphRecursionError
except ImportError:  # older/newer langgraph layouts
    class GraphRecursionError(Exception):
        pass


def approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "rate_limit" in msg
        or "request too large" in msg
        or "tokens per minute" in msg
        or "error code: 429" in msg
        or "error code: 413" in msg
    )


def invoke_with_retry(runnable, payload, *, config=None, label="call", max_retries=3, cooldown=65):
    """
    Invoke a LangChain runnable, waiting out Groq's per-minute token window
    (ITPM) if it rejects the request with a 413/429 rate-limit error.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return runnable.invoke(payload, config) if config else runnable.invoke(payload)
        except Exception as err:  # noqa: BLE001 - we re-raise anything unexpected
            if not _is_rate_limit_error(err) or attempt == max_retries:
                raise
            print(
                f"[rate-limit] {label}: hit Groq ITPM cap, "
                f"waiting {cooldown}s for the window to clear "
                f"(attempt {attempt}/{max_retries - 1})"
            )
            time.sleep(cooldown)


def invoke_agent_text(agent, payload, *, config=None, label="agent", fallback="", **retry_kw):
    """
    Run a `create_agent` graph and return its final message as plain text.

    If the agent gets stuck in a tool-calling loop and hits the recursion
    limit, return `fallback` instead of raising GraphRecursionError so the
    pipeline can keep going with whatever it already has.
    """
    try:
        result = invoke_with_retry(agent, payload, config=config, label=label, **retry_kw)
    except GraphRecursionError:
        print(f"[recursion] {label}: hit the step limit, falling back")
        return fallback

    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        return fallback
    return str(getattr(messages[-1], "content", "") or fallback)
