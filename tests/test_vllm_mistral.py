"""Check the vLLM stream adapter without a running CUDA server."""

from contextlib import nullcontext
import json
from threading import local
from types import SimpleNamespace

import pytest

pytest.importorskip("requests")

from experiments.vllm_mistral import VLLMMistral


@pytest.mark.parametrize("stop_at_json", [True, False])
def test_stream_response_uses_the_pinned_template_and_reports_token_count(stop_at_json):
    chunks = [
        {"choices": [{"text": '{"label":', "token_ids": [7, 8]}]},
        {"choices": [{"text": '"present"}', "token_ids": [9, 10]}]},
        {"choices": [{"text": " trailing text", "token_ids": [11, 12], "finish_reason": "length"}]},
        {"choices": [], "usage": {"completion_tokens": 6}},
    ]
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            iter_lines=lambda **kwargs: (
                b"data: " + json.dumps(chunk).encode() for chunk in chunks
            ),
        )
        return nullcontext(response)

    model = VLLMMistral.__new__(VLLMMistral)
    model.url = "http://127.0.0.1:8000/v1"
    model.tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: f"[INST] {messages[0]['content']} [/INST]",
        encode=lambda prompt, **kwargs: [1, 2, 3],
    )
    model.input_limit, model.output_limit, model.seed = 8, 16, 42
    model.stop_token_ids = [2]
    model.worker = local()
    model.worker.session = SimpleNamespace(post=post)

    result = model.generate_for_baseline("Question", stop_at_json=stop_at_json)

    assert result["model_prompt"] == "[INST] Question [/INST]"
    assert result["text"] == '{"label":"present"}' + ("" if stop_at_json else " trailing text")
    assert result["tokens"] == (4 if stop_at_json else 6)
    assert model.calls[0]["finish_reason"] == ("json_object" if stop_at_json else "length")
    url, request = calls[0]
    assert url.endswith("/completions")
    assert request["json"] == {
        "model": "mistral-audit", "prompt": [1, 2, 3], "max_tokens": 16, "temperature": 0, "seed": 42,
        "stop_token_ids": [2], "stream": True, "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
