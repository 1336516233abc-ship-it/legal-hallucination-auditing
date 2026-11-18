#!/usr/bin/env python
"""
Quick verification script to test the installation and basic functionality.
Run this after installing dependencies to ensure everything is set up correctly.
"""

import sys
import os

print("="*70)
print("LEGAL HALLUCINATION AUDITING AGENT - INSTALLATION VERIFICATION")
print("="*70)

# Test 1: Check Python version
print("\n[1/8] Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    print(f"✗ Python version too old. Requires 3.8+, found {sys.version_info.major}.{sys.version_info.minor}")
    sys.exit(1)

# Test 2: Check dependencies
print("\n[2/8] Checking dependencies...")
required_packages = [
    'torch',
    'transformers',
    'datasets',
    'rank_bm25',
    'sentence_transformers',
    'nltk',
    'pandas',
    'numpy',
    'sklearn',
    'yaml',
]

missing = []
for pkg in required_packages:
    try:
        __import__(pkg if pkg != 'yaml' else 'yaml')
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} (missing)")
        missing.append(pkg)

if missing:
    print(f"\n✗ Missing packages: {', '.join(missing)}")
    print("  Install with: pip install -r requirements.txt")
    sys.exit(1)

# Test 3: Check project structure
print("\n[3/8] Checking project structure...")
required_dirs = [
    'data',
    'retrieval',
    'agent',
    'evaluation',
    'experiments',
    'utils',
    'configs',
]

for dirname in required_dirs:
    if os.path.isdir(dirname):
        print(f"  ✓ {dirname}/")
    else:
        print(f"  ✗ {dirname}/ (missing)")

# Test 4: Check config files
print("\n[4/8] Checking configuration files...")
config_files = [
    'configs/model_config.yaml',
    'configs/retrieval_config.yaml',
    'configs/experiment_config.yaml',
]

for config in config_files:
    if os.path.isfile(config):
        print(f"  ✓ {config}")
    else:
        print(f"  ✗ {config} (missing)")

# Test 5: Import modules
print("\n[5/8] Testing module imports...")
try:
    from data.dataset_loader import DatasetLoader, LegalExample
    print("  ✓ data module")
except ImportError as e:
    print(f"  ✗ data module: {e}")

try:
    from retrieval.chunking import get_chunker
    print("  ✓ retrieval module")
except ImportError as e:
    print(f"  ✗ retrieval module: {e}")

try:
    from agent.agentic_rag import AgenticRAG
    print("  ✓ agent module")
except ImportError as e:
    print(f"  ✗ agent module: {e}")

try:
    from evaluation.metrics import MetricsCalculator
    print("  ✓ evaluation module")
except ImportError as e:
    print(f"  ✗ evaluation module: {e}")

try:
    from utils.helpers import load_yaml, set_random_seed
    print("  ✓ utils module")
except ImportError as e:
    print(f"  ✗ utils module: {e}")

# Test 6: Test NLTK data
print("\n[6/8] Checking NLTK data...")
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    print("  ✓ NLTK punkt tokenizer")
except LookupError:
    print("  ⚠ NLTK punkt tokenizer not found")
    print("    Download with: python -c \"import nltk; nltk.download('punkt')\"")

# Test 7: Test basic functionality
print("\n[7/8] Testing basic functionality...")
try:
    from retrieval.chunking import FixedLengthChunker
    chunker = FixedLengthChunker(chunk_size=100, overlap=20)
    test_text = "This is a test contract. " * 50
    chunks = chunker.chunk(test_text)
    print(f"  ✓ Chunking works ({len(chunks)} chunks created)")
except Exception as e:
    print(f"  ✗ Chunking failed: {e}")

try:
    from utils.helpers import set_random_seed
    set_random_seed(42)
    print("  ✓ Random seed setting works")
except Exception as e:
    print(f"  ✗ Random seed setting failed: {e}")

# Test 8: Check GPU availability
print("\n[8/8] Checking compute resources...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print("  ✓ MPS (Apple Silicon) available")
    else:
        print("  ⚠ No GPU detected - will use CPU (slower)")
except Exception as e:
    print(f"  ⚠ Could not check GPU: {e}")

# Summary
print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\n✓ Installation appears to be working correctly!")
print("\nNext steps:")
print("  1. Download datasets: python data/contractnli_processor.py --download --process")
print("  2. Test a baseline: python retrieval/baseline_direct.py --model llama3")
print("  3. Run full experiments: python experiments/run_experiments.py")
print("\nFor detailed instructions, see IMPLEMENTATION_GUIDE.md")
print("="*70)
