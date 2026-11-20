"""Call a pinned checkpoint through a local vLLM completion server."""

import json
from pathlib import Path
from threading import local
from urllib.parse import urlsplit

import requests

from retrieval.common import completed_json_object


SERVED_NAME = "mistral-audit"


class VLLMMistral:
    """Render the model template once and keep generation traces local to each worker."""

    served_name = SERVED_NAME

    def __init__(self, snapshot, url, input_limit, output_limit, seed, *, served_name=SERVED_NAME):
        from transformers import AutoTokenizer

        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Use a loopback HTTP URL for the local vLLM server")
        self.url = url.rstrip("/")
        self.served_name = served_name
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.config = json.loads((snapshot / "config.json").read_text())
        generation_config = json.loads((snapshot / "generation_config.json").read_text())
        eos = generation_config.get("eos_token_id", self.tokenizer.eos_token_id)
        self.stop_token_ids = [eos] if isinstance(eos, int) else eos
        self.baseline_tokenizer = self
        self.input_limit, self.output_limit, self.seed = input_limit, output_limit, seed
        self.worker = local()
        with requests.get(f"{self.url}/models", timeout=15) as response:
            response.raise_for_status()
            models = response.json()["data"]
        if not any(model["id"] == self.served_name and Path(model["root"]).resolve() == snapshot.resolve()
                   for model in models):
            raise ValueError("vLLM is not serving the requested pinned snapshot")

    @property
    def calls(self):
        if not hasattr(self.worker, "calls"):
            self.worker.calls = []
        return self.worker.calls

    def _format(self, text):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                  tokenize=False, add_generation_prompt=True)

    def encode(self, text, add_special_tokens=True):
        return self.tokenizer.encode(self._format(text), add_special_tokens=False)

    def generate_for_baseline(self, text, max_tokens=None, temperature=0, *, stop_at_json=False):
        # Match the pilot's greedy decoding for all Agent stages.
        prompt = self._format(text)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        limit = self.output_limit if max_tokens is None else max_tokens
        if len(tokens) > self.input_limit or not 0 < limit <= self.output_limit:
            raise ValueError("Generation request exceeds the locked experiment budget")
        payload = {"model": self.served_name, "prompt": tokens, "max_tokens": limit, "temperature": 0,
                   "seed": self.seed, "stop_token_ids": self.stop_token_ids, "stream": True,
                   "return_token_ids": True, "stream_options": {"include_usage": True}}
        if not hasattr(self.worker, "session"):
            self.worker.session = requests.Session()
        output, reason, output_tokens = "", None, 0
        with self.worker.session.post(f"{self.url}/completions", json=payload, stream=True,
                                      timeout=(10, 240)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=128):
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(f"vLLM stream failed: {event['error']}")
                if event.get("usage"):
                    output_tokens = event["usage"]["completion_tokens"]
                for choice in event.get("choices", []):
                    output += choice.get("text", "")
                    output_tokens += len(choice.get("token_ids") or [])
                    reason = choice.get("finish_reason") or reason
                completed = completed_json_object(output) if stop_at_json else None
                if completed is not None:
                    output, reason = completed, "json_object"
                    break
        if reason is None:
            raise RuntimeError("vLLM stream closed without a completion reason")
        if output and not output_tokens:
            raise RuntimeError("vLLM did not report generated token IDs or usage")
        result = {"text": output.strip(), "tokens": output_tokens, "model_prompt": prompt}
        self.calls.append({"input_tokens": len(tokens), "max_output_tokens": limit,
                           "finish_reason": reason, **result})
        return result

    generate = generate_for_baseline

    def generate_json_for_baseline(self, prompt):
        return self.generate_for_baseline(prompt, stop_at_json=True)
