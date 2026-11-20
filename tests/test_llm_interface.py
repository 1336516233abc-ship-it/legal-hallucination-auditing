"""Optional backend regressions; no model weights are loaded by these tests."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from utils.llm_interface import LLMInterface


def test_mps_selection_and_cpu_fallback(monkeypatch):
    monkeypatch.setattr(LLMInterface, "_init_local_model", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert LLMInterface("unused", device="mps").device == "mps"
    assert LLMInterface("unused", device="cuda").device == "cpu"


def test_zero_temperature_is_not_replaced_by_default():
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1
        def __call__(self, prompt, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}
        def decode(self, ids, **kwargs):
            return "generated"
    class Model:
        generation_config = SimpleNamespace(eos_token_id=[1, 4])

        def generate(self, **kwargs):
            assert kwargs["temperature"] == 0
            assert kwargs["do_sample"] is False
            assert kwargs["eos_token_id"] == [1, 4]
            return torch.tensor([[1, 2, 3]])
    llm = LLMInterface.__new__(LLMInterface)
    llm.tokenizer, llm.model = Tokenizer(), Model()
    llm.device, llm.max_tokens, llm.temperature, llm.top_p = "cpu", 10, .7, .9
    result = llm._generate_local("prompt", temperature=0)
    assert result["text"] == "generated" and result["tokens"] == 1


def test_fp16_mps_load_places_weights_directly_on_device(monkeypatch):
    from utils import llm_interface

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    tokenizer = SimpleNamespace(pad_token="[PAD]")
    monkeypatch.setattr(llm_interface.AutoTokenizer, "from_pretrained", lambda name: tokenizer)

    def load(name, **kwargs):
        assert kwargs["device_map"] == {"": "mps"}
        assert kwargs["torch_dtype"] is torch.float16
        return SimpleNamespace(eval=lambda: None)

    monkeypatch.setattr(llm_interface.AutoModelForCausalLM, "from_pretrained", load)
    assert LLMInterface("unused", device="mps").device == "mps"


def test_baseline_template_budget_and_generation_share_exact_text(monkeypatch):
    class Tokenizer:
        chat_template = "available"
        def apply_chat_template(self, messages, **kwargs):
            return "<user>" + messages[0]["content"] + "<assistant>"
        def encode(self, text, add_special_tokens=True):
            assert not add_special_tokens
            return list(text)
    llm = LLMInterface.__new__(LLMInterface)
    llm.tokenizer = Tokenizer()
    def generate(prompt, **kwargs):
        assert prompt == "<user>question<assistant>"
        assert kwargs["add_special_tokens"] is False
        return {"text": "answer", "tokens": 1}
    monkeypatch.setattr(llm, "generate", generate)
    assert len(llm.baseline_tokenizer.encode("question")) == len("<user>question<assistant>")
    assert llm.generate_for_baseline("question")["model_prompt"] == "<user>question<assistant>"


def test_local_json_boundary_uses_only_generated_tokens():
    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def __call__(self, prompt, **kwargs):
            return {"input_ids": torch.tensor([[2, 3]])}

        def decode(self, ids, **kwargs):
            lookup = {2: '{"in_prompt":', 3: 'true}', 4: '{"label":"absent"', 5: '} trailing'}
            return "".join(lookup[int(token)] for token in ids)

    class Model:
        def generate(self, **kwargs):
            criterion = kwargs["stopping_criteria"][0]
            assert not criterion(torch.tensor([[2, 3, 4]]), None)
            ids = torch.tensor([[2, 3, 4, 5]])
            assert criterion(ids, None)
            return ids

    llm = LLMInterface.__new__(LLMInterface)
    llm.tokenizer, llm.model = Tokenizer(), Model()
    llm.device, llm.max_tokens, llm.temperature, llm.top_p = "cpu", 10, 0, .9
    assert llm._generate_local("prompt", stop_at_json=True) == {
        "text": '{"label":"absent"}', "tokens": 2, "finish_reason": "json_object",
    }
