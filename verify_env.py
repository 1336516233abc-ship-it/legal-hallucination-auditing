"""
Environment Setup Verification - Wang Zhengyao
Verify all necessary dependencies and configurations
"""

import sys
import os

print("=" * 60)
print("Environment Setup Verification - Wang Zhengyao")
print("=" * 60)

# Check project directory
print("\n1. Check project directory...")
project_dir = os.getcwd()
print(f"   Project directory: {project_dir}")
assert "legal-hallucination-auditing" in project_dir, "Not in correct project directory"
print("   [OK] Project directory correct")

# Check Python version
print("\n2. Check Python version...")
python_version = sys.version_info
print(f"   Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
assert python_version.major == 3 and python_version.minor >= 8, "Python version must be >= 3.8"
print("   [OK] Python version meets requirements")

# Check required packages
print("\n3. Check required Python packages...")
required_packages = {
    'openai': 'OpenAI SDK (for OpenRouter)',
    'groq': 'Groq SDK (backup)',
    'dotenv': 'python-dotenv (env management)',
}

all_packages_ok = True
for package, description in required_packages.items():
    try:
        if package == 'dotenv':
            __import__('dotenv')
        else:
            __import__(package)
        print(f"   [OK] {package}: {description}")
    except ImportError:
        print(f"   [FAIL] {package} not installed: {description}")
        print(f"          Run: pip install {package}")
        all_packages_ok = False

# Check API Key
print("\n4. Check API Key configuration...")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")
if api_key:
    print(f"   [OK] API Key configured: {api_key[:30]}...")
else:
    print("   [FAIL] API Key not configured")
    print("          Set OPENROUTER_API_KEY in .env file")

# Check project structure
print("\n5. Check project structure...")
required_dirs = [
    'agent',
    'agent/query_analyzer',
    'agent/verifier',
    'agent/reviser',
    'utils',
    'data',
    'retrieval',
    'evaluation',
    'experiments',
]

for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print(f"   [OK] {dir_name}/")
    else:
        print(f"   [MISS] {dir_name}/ does not exist")

# Test OpenRouter API
print("\n6. Test OpenRouter API connection...")
try:
    # Direct module loading to avoid __init__.py dependency issues
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "openrouter_llm",
        "utils/openrouter_llm.py"
    )
    openrouter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(openrouter_module)

    OpenRouterLLM = openrouter_module.OpenRouterLLM

    llm = OpenRouterLLM(model_name="llama3", max_tokens=50)
    print("   [OK] OpenRouter LLM initialized successfully")

    # Simple test
    result = llm.generate("Say 'OK' if you can read this.", max_tokens=10)
    print(f"   [OK] API call successful: {result['text'][:50]}...")
    print(f"        Tokens: {result['tokens']}, Time: {result['latency']:.2f}s")

except Exception as e:
    print(f"   [FAIL] OpenRouter test failed: {str(e)}")
    print("          Check if API Key is correct")

# Check Agent component files
print("\n7. Check Agent component files...")
agent_files = [
    'agent/query_analyzer/analyzer.py',
    'agent/generator.py',
    'agent/verifier/verifier.py',
    'agent/reviser/reviser.py',
    'agent/agentic_rag.py',
]

for file_path in agent_files:
    if os.path.exists(file_path):
        print(f"   [OK] {file_path}")
    else:
        print(f"   [TODO] {file_path} (needs implementation)")

print("\n" + "=" * 60)
print("Environment Setup Verification Complete!")
print("=" * 60)

print("\nNext steps:")
print("1. Implement Agent components (query_analyzer, verifier, reviser)")
print("2. Run small sample tests")
print("3. Integrate into complete Agentic RAG workflow")

print("\nQuick test commands:")
print("  python verify_setup.py")
print("  python test_final.py")
