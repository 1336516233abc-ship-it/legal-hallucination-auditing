from setuptools import setup, find_packages

setup(
    name="legal-hallucination-auditing",
    version="0.1.0",
    description="Agentic RAG system for auditing hallucinations in legal contract analysis",
    author="Team: Li Xinru, Qin Mengmeng, Yin Fengling, Wang Zhengyao",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.26.0",
        "sentence-transformers>=2.2.0",
        "datasets>=2.12.0",
        "rank-bm25>=0.2.2",
        "faiss-cpu>=1.7.4",
        "nltk>=3.8",
        "spacy>=3.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "rouge-score>=0.1.2",
        "evaluate>=0.4.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "jsonlines>=3.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
        "ui": [
            "gradio>=3.35.0",
            "streamlit>=1.24.0",
        ],
    },
)
