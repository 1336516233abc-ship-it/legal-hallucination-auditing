"""
Unified LLM interface for multiple model backends.
Supports Llama3, Mistral, and other models via HuggingFace or API.

Author: Shared utility
"""

import os
import time
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Optional imports for local model inference
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("PyTorch/Transformers not available. Only API mode will work.")


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
        self.device = device if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) or device == "cpu" else "cpu"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_api = use_api
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY")

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
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "PyTorch and Transformers are required for local model inference. "
                "Install with: pip install torch transformers"
            )

        try:
            logger.info(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True,
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

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

        # Initialize Groq API client
        try:
            from groq import Groq
            self.api_client = Groq(api_key=self.api_key)

            # Map model names to Groq model IDs
            self.model_map = {
                "llama3": "llama3-8b-8192",
                "llama3-8b": "llama3-8b-8192",
                "llama3-70b": "llama3-70b-8192",
                "mixtral": "mixtral-8x7b-32768",
                "mistral": "mixtral-8x7b-32768",
            }

            # Get Groq model ID
            self.groq_model = self.model_map.get(
                self.model_name.lower(),
                "llama3-8b-8192"  # Default to Llama3-8B
            )

            logger.info(f"Groq API initialized with model: {self.groq_model}")

        except ImportError:
            logger.error("Groq package not installed. Run: pip install groq")
            raise
        except Exception as e:
            logger.error(f"Error initializing Groq API: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
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
            result = self._generate_local(prompt, max_tokens, temperature, stop_sequences)

        latency = time.time() - start_time
        result["latency"] = latency

        # Track tokens
        self.total_tokens += result.get("tokens", 0)

        return result

    def _generate_local(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict:
        """Generate using local model."""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Count tokens
        num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        return {
            "text": generated_text.strip(),
            "tokens": num_tokens,
        }

    def _generate_api(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict:
        """Generate using Groq API."""
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        try:
            response = self.api_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
                stop=stop_sequences,
            )

            generated_text = response.choices[0].message.content
            num_tokens = response.usage.total_tokens

            return {
                "text": generated_text.strip(),
                "tokens": num_tokens,
            }

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

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
