"""Deterministic model doubles for interface and boundary tests."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


class CharacterTokenizer:
    model_max_length = 100000

    def encode(self, text, add_special_tokens=True):
        return list(text)


class RecordingLLM:
    def __init__(self, response=None, **kwargs):
        self.prompts = []
        self.tokenizer = CharacterTokenizer()
        self.response = response

    def generate(self, prompt):
        self.prompts.append(prompt)
        text = self.response
        if text is None:
            text = json.dumps({"answer": "", "label": "absent"} if '"extracted_spans"' in prompt
                              else {"answer": "Not specified", "label": "neutral"})
        return {"text": text, "tokens": 5}


@pytest.fixture
def llm():
    return RecordingLLM()


@pytest.fixture
def tiny_data(tmp_path):
    data = [dict(id=f"nli_{i}", query="Is payment required?", context=f"Contract {i}. Payment is due.",
                 label="entailment", answer=None, evidence_spans=[[12, 19]],
                 metadata={"dataset": "contractnli", "document_id": i}) for i in range(12)]
    (tmp_path / "contractnli_test.json").write_text(json.dumps(data))
    return tmp_path
