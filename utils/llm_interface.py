"""
Unified LLM interface for multiple model backends.
Supports Llama3, Mistral, and other models via HuggingFace or API.

Author: Shared utility
"""

import os
import time
from typing import List, Dict, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, pipeline
import logging
from retrieval.common import completed_json_object

logger = logging.getLogger(__name__)


class _BaselineTokenizer:
    """Count the exact formatted prompt seen by the local generation model."""

    def __init__(self, llm):
        self.llm = llm

    def encode(self, text, add_special_tokens=True):
        prompt, templated = self.llm._format_baseline_prompt(text)
        return self.llm.tokenizer.encode(prompt, add_special_tokens=not templated)


class _StopAtJSONObject(StoppingCriteria):
    """Stop single-prompt decoding when the generated JSON object closes."""

    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        emitted = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
        return completed_json_object(emitted) is not None


class LLMInterface:
    """Unified interface for language model inference."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_api: bool = False,
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM interface.

        Args:
            model_name: Model identifier (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
            device: Device to run on ("cuda", "cpu", "mps")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_api: Whether to use API instead of local model
            api_key: API key for hosted models
        """
        self.model_name = model_name
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_api = use_api
        self.api_key = api_key or os.getenv("LLM_API_KEY")

        # Token tracking
        self.total_tokens = 0

        # Initialize model
        if use_api:
            self._init_api()
        else:
            self._init_local_model()

        logger.info(f"Initialized LLM: {model_name} on {self.device}")

    def _init_local_model(self):
        """Initialize local model via HuggingFace."""
        try:
            logger.info(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Place weights directly on MPS to avoid holding a second full model in memory.
            device_maps = {"cuda": "auto", "mps": {"": "mps"}}
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
                device_map=device_maps.get(self.device),
                low_cpu_mem_usage=True,
            )

            self.model.eval()

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _init_api(self):
        """Initialize API client for hosted models."""
        logger.info(f"Using API for model {self.model_name}")
        # Placeholder for API initialization
        # Implement based on specific API (OpenAI, Anthropic, etc.)
        pass

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        add_special_tokens: bool = True,
        *,
        stop_at_json: bool = False,
    ) -> Dict:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stop_sequences: List of stop sequences

        Returns:
            Dict with 'text', 'tokens', 'latency'
        """
        start_time = time.time()

        if self.use_api:
            result = self._generate_api(prompt, max_tokens, temperature, stop_sequences)
        else:
            result = self._generate_local(prompt, max_tokens, temperature, stop_sequences, add_special_tokens,
                                          stop_at_json=stop_at_json)

        latency = time.time() - start_time
        result["latency"] = latency

        # Track tokens
        self.total_tokens += result.get("tokens", 0)

        return result

    @property
    def baseline_tokenizer(self):
        return _BaselineTokenizer(self)

    def _format_baseline_prompt(self, prompt):
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True), True
        return prompt, False

    def generate_for_baseline(self, prompt, **generation_options):
        """Use the model's chat template without changing legacy agent calls."""
        formatted, templated = self._format_baseline_prompt(prompt)
        result = self.generate(formatted, add_special_tokens=not templated, **generation_options)
        result["model_prompt"] = formatted
        return result

    def generate_json_for_baseline(self, prompt):
        """Generate a task response with a single JSON object boundary."""
        return self.generate_for_baseline(prompt, stop_at_json=True)

    def _generate_local(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        add_special_tokens: bool = True,
        *,
        stop_at_json: bool = False,
    ) -> Dict:
        """Generate using local model."""
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        if max_tokens <= 0 or temperature < 0:
            raise ValueError("max_tokens must be positive and temperature nonnegative")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Keep the checkpoint's end-of-turn tokens for chat models.
        eos_token_id = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        prompt_length = inputs["input_ids"].shape[1]
        options = {}
        if stop_at_json:
            options["stopping_criteria"] = StoppingCriteriaList([_StopAtJSONObject(self.tokenizer, prompt_length)])
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                **options,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True
        )

        # Count tokens
        num_tokens = outputs.shape[1] - prompt_length

        completed = completed_json_object(generated_text) if stop_at_json else None
        result = {"text": completed if completed is not None else generated_text.strip(), "tokens": num_tokens}
        if completed is not None:
            result["finish_reason"] = "json_object"
        return result

    def _generate_api(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict:
        """Generate using API."""
        # Placeholder for API generation
        # Implement based on specific API
        raise NotImplementedError("API generation not implemented yet")

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        Chat completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            Dict with 'text', 'tokens', 'latency'
        """
        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant:"

        return self.generate(prompt, max_tokens, temperature)

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[Dict]:
        """
        Generate for multiple prompts in batch.

        Args:
            prompts: List of prompts
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            List of generation results
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens, temperature)
            results.append(result)

        return results

    def get_token_count(self) -> int:
        """Get total tokens used."""
        return self.total_tokens

    def reset_token_count(self):
        """Reset token counter."""
        self.total_tokens = 0


def load_model_from_config(config: Dict) -> LLMInterface:
    """
    Load model from configuration dict.

    Args:
        config: Configuration dictionary

    Returns:
        LLMInterface instance
    """
    return LLMInterface(
        model_name=config.get("name", "meta-llama/Meta-Llama-3-8B-Instruct"),
        device=config.get("device", "cuda"),
        max_tokens=config.get("max_tokens", 512),
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        use_api=config.get("use_api", False),
        api_key=config.get("api_key"),
    )
