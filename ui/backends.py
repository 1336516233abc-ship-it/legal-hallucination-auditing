"""Local model loading for CUDA/CPU/MPS and the existing Apple Silicon backend."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[1]


class ThreadBoundModel:
    """Create and run a thread-affine backend on the same dedicated worker."""

    def __init__(self, factory, *args, **kwargs):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-model")
        try:
            self.model = self.executor.submit(factory, *args, **kwargs).result()
        except BaseException:
            self.executor.shutdown(wait=True)
            raise

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate_for_baseline(self, prompt, **options):
        return self.executor.submit(self.model.generate_for_baseline, prompt, **options).result()

    def close(self):
        self.executor.shutdown(wait=True)


class BoundedChatModel:
    """Apply the backend chat template and input budget to every Agent stage."""

    def __init__(self, model, max_input_tokens):
        self.model = model
        self.max_input_tokens = max_input_tokens
        self.tokenizer = getattr(model, "baseline_tokenizer", model.tokenizer)
        self.baseline_tokenizer = self.tokenizer

    def generate(self, prompt, **options):
        if len(self.tokenizer.encode(prompt, add_special_tokens=True)) > self.max_input_tokens:
            raise ValueError("Model prompt exceeds the configured input token budget")
        return self.model.generate_for_baseline(prompt, **options)

    generate_for_baseline = generate

    def generate_json_for_baseline(self, prompt):
        return self.generate(prompt, stop_at_json=True)


def load_backend(model_name, backend, snapshot, device, max_input_tokens):
    import yaml
    config = yaml.safe_load((ROOT / "configs/model_config.yaml").read_text())
    source = config["model"][model_name]["name"]
    info = {"model": model_name, "backend": backend, "temperature": 0, "seed": 42,
            "max_input_tokens": max_input_tokens, "max_output_tokens": 512}
    if snapshot is not None:
        snapshot = Path(snapshot).expanduser().resolve()
        model_config = json.loads((snapshot / "config.json").read_text())
        expected = {"llama3": "llama", "mistral": "mistral"}[model_name]
        if model_config.get("model_type") != expected:
            raise ValueError(f"--model {model_name} does not match the local model family")
        source = str(snapshot)
        info["checkpoint"] = "local snapshot"
    else:
        info["checkpoint"] = source
    if backend == "mlx":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("MLX requires Apple Silicon; use --backend transformers on this machine")
        if snapshot is None:
            raise ValueError("MLX requires --snapshot with local model weights")
        if device != "auto":
            raise ValueError("MLX uses Metal automatically; omit --device")
        from experiments.run_baseline_pilot import MLXPilotModel
        # MLX streams and lazily loaded arrays must be created and used on one worker.
        model = ThreadBoundModel(MLXPilotModel, snapshot, record_calls=False)
        context_limit = model.config["max_position_embeddings"]
        info["device"] = "metal"
    else:
        import torch
        from transformers import set_seed
        from utils.llm_interface import LLMInterface
        available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
        if device == "auto":
            device = next(name for name in ("cuda", "mps", "cpu") if available[name])
        if not available[device]:
            raise ValueError(f"Requested device {device} is unavailable")
        set_seed(42)
        model = LLMInterface(source, device=device, max_tokens=512, temperature=0)
        context_limit = model.model.config.max_position_embeddings
        info["device"] = device
    if max_input_tokens + 512 > context_limit:
        raise ValueError("The input and output token budgets exceed this model's context window")
    return BoundedChatModel(model, max_input_tokens), info
