"""
Helper utilities for the project.

Author: Shared utility
"""

import random
import numpy as np
import torch
import yaml
import json
from typing import Dict, Any, List
import os


def set_random_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_yaml(filepath: str) -> Dict:
    """Load YAML configuration file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, filepath: str):
    """Save data to YAML file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(directory: str):
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    """
    Chunk text into overlapping segments.

    Args:
        text: Input text
        chunk_size: Size of each chunk in tokens (approximate)
        overlap: Overlap between chunks in tokens (approximate)

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with variables.

    Args:
        template: Prompt template with {variable} placeholders
        **kwargs: Variables to fill in

    Returns:
        Formatted prompt
    """
    return template.format(**kwargs)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def compute_overlap(span1: tuple, span2: tuple) -> int:
    """
    Compute character overlap between two spans.

    Args:
        span1: Tuple of (start, end)
        span2: Tuple of (start, end)

    Returns:
        Number of overlapping characters
    """
    start1, end1 = span1
    start2, end2 = span2

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    return max(0, overlap_end - overlap_start)


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Recursively merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def get_device() -> str:
    """
    Get the best available device.

    Returns:
        Device string ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
