"""Token usage from raw, provider-shaped model responses.

`agent_messages` (computer_use_turns), `events.data.output` and
`claude_code_messages.content` store the provider response as-is. This module
finds the usage block in one and normalizes it to four billable buckets:

- ``input``: uncached input tokens, billed at the full input rate
- ``cache_read``: input tokens served from the prompt cache
- ``cache_write``: input tokens written to the cache (Anthropic only)
- ``output``: output tokens, including reasoning/thinking tokens

Providers count these differently. Anthropic's ``input_tokens`` excludes
cache reads and writes, while OpenAI's and Gemini's prompt counts include the
cached part. Gemini's ``candidatesTokenCount`` excludes thinking tokens, which
are billed as output.
"""

from dataclasses import dataclass
from typing import Any

_USAGE_KEYS = ("usage", "usageMetadata", "usage_metadata")
_MODEL_KEYS = ("model", "modelVersion")
_MAX_DEPTH = 6


@dataclass(frozen=True)
class Usage:
    input: int
    cache_read: int
    cache_write: int
    output: int
    provider: str  # "anthropic", "openai" or "gemini": the usage block's format

    @property
    def total_input(self) -> int:
        return self.input + self.cache_read + self.cache_write


def _find(obj: Any, keys: tuple[str, ...], want: type, depth: int = 0) -> Any:
    """Depth-first search for the first value under one of `keys` of type `want`."""
    if depth > _MAX_DEPTH:
        return None
    if isinstance(obj, dict):
        for k in keys:
            if isinstance(obj.get(k), want):
                return obj[k]
        children = obj.values()
    elif isinstance(obj, list):
        children = obj
    else:
        return None
    for child in children:
        found = _find(child, keys, want, depth + 1)
        if found is not None:
            return found
    return None


def _int(d: dict, *path: str) -> int:
    for key in path:
        if not isinstance(d, dict):
            return 0
        d = d.get(key)
    return d if isinstance(d, int) else 0


def normalize(u: dict) -> Usage | None:
    """Normalize one usage block, or None if its format isn't recognized."""
    if "promptTokenCount" in u or "candidatesTokenCount" in u:
        prompt = _int(u, "promptTokenCount")
        cached = _int(u, "cachedContentTokenCount")
        output = _int(u, "candidatesTokenCount") + _int(u, "thoughtsTokenCount")
        return Usage(prompt - cached, cached, 0, output, "gemini")
    if "cache_read_input_tokens" in u or "cache_creation_input_tokens" in u:
        return Usage(
            _int(u, "input_tokens"),
            _int(u, "cache_read_input_tokens"),
            _int(u, "cache_creation_input_tokens"),
            _int(u, "output_tokens"),
            "anthropic",
        )
    if "prompt_tokens" in u:  # OpenAI chat completions and compatible APIs
        prompt = _int(u, "prompt_tokens")
        cached = _int(u, "prompt_tokens_details", "cached_tokens") or _int(
            u, "prompt_cache_hit_tokens"
        )
        return Usage(prompt - cached, cached, 0, _int(u, "completion_tokens"), "openai")
    if "input_tokens" in u:  # OpenAI Responses API
        total = _int(u, "input_tokens")
        cached = _int(u, "input_tokens_details", "cached_tokens")
        return Usage(total - cached, cached, 0, _int(u, "output_tokens"), "openai")
    return None


def extract(response: Any) -> Usage | None:
    """Find and normalize the usage block in a raw model response, if any."""
    u = _find(response, _USAGE_KEYS, dict)
    return normalize(u) if u is not None else None


def response_id(response: Any) -> str | None:
    """The provider's id for a raw response (Anthropic ``msg_…``, Gemini ``responseId``)."""
    if isinstance(response, dict) and isinstance(response.get("message"), dict):
        response = response["message"]  # Claude Code SDK entries wrap the API message
    return _find(response, ("responseId", "id"), str)


def model(response: Any) -> str | None:
    """The model name a raw response reports, if any (e.g. ``modelVersion``)."""
    m = _find(response, _MODEL_KEYS, str)
    return m.removeprefix("models/") if m else None
