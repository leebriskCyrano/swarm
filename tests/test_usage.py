import pytest

from aivillage import usage
from aivillage.usage import Usage

ANTHROPIC = {
    "id": "msg_1",
    "model": "claude-opus-4-5-20251101",
    "content": [],
    "usage": {
        "input_tokens": 10,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 50,
        "output_tokens": 40,
    },
}
GEMINI = {
    "candidates": [],
    "modelVersion": "models/gemini-2.5-pro",
    "usageMetadata": {
        "promptTokenCount": 1000,
        "cachedContentTokenCount": 600,
        "candidatesTokenCount": 30,
        "thoughtsTokenCount": 70,
    },
}
OPENAI_CHAT = {
    "model": "gpt-5",
    "usage": {
        "prompt_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": 200},
        "completion_tokens": 25,
    },
}
OPENAI_RESPONSES = {
    "usage": {
        "input_tokens": 800,
        "input_tokens_details": {"cached_tokens": 300},
        "output_tokens": 9,
    }
}


@pytest.mark.parametrize(
    "response, expected",
    [
        (ANTHROPIC, Usage(10, 900, 50, 40, "anthropic")),
        (GEMINI, Usage(400, 600, 0, 100, "gemini")),
        (OPENAI_CHAT, Usage(300, 200, 0, 25, "openai")),
        (OPENAI_RESPONSES, Usage(500, 300, 0, 9, "openai")),
        # usage nested inside an SDK wrapper
        (
            {"_sdkFormat": 1, "textMessage": {"message": ANTHROPIC}},
            Usage(10, 900, 50, 40, "anthropic"),
        ),
    ],
)
def test_extract(response, expected):
    assert usage.extract(response) == expected


def test_extract_missing():
    assert usage.extract([{"type": "reasoning"}, {"type": "function_call"}]) is None
    assert usage.extract(None) is None
    assert usage.extract({"usage": {"unknown": 1}}) is None


def test_total_input():
    assert usage.extract(ANTHROPIC).total_input == 960


def test_model():
    assert usage.model(ANTHROPIC) == "claude-opus-4-5-20251101"
    assert usage.model(GEMINI) == "gemini-2.5-pro"
    assert usage.model([{"type": "message"}]) is None
