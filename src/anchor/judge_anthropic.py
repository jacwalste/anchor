"""Anthropic-backed judge client. Requires the `anthropic` extra.

Core never imports this module — the SDK dependency only loads when someone
actually constructs an AnthropicJudge. There is no default model id: the
judge is always chosen explicitly (D4 — never hardcode a judge).
"""

try:
    from anthropic import Anthropic
except ImportError as exc:
    raise ImportError(
        "AnthropicJudge requires the 'anthropic' extra: pip install anchor[anthropic]"
    ) from exc


class AnthropicJudge:
    """JudgeClient backed by the Anthropic Messages API.

    Credentials come from the environment (ANTHROPIC_API_KEY) unless a
    configured client is injected. Thread-safe — usable with
    evaluate_answer's max_concurrency.
    """

    def __init__(self, model: str, max_tokens: int = 4096, client: Anthropic | None = None) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = client if client is not None else Anthropic()

    def complete(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        texts = [block.text for block in response.content if block.type == "text"]
        if not texts:
            raise ValueError(f"judge model {self._model!r} returned no text blocks")
        return "\n".join(texts)
