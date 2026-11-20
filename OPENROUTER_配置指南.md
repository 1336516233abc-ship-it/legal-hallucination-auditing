# 🚀 OpenRouter API 配置指南 - 王正尧

## ✅ 为什么选择 OpenRouter？

✅ **香港可访问** - 不会被墙，稳定可靠
✅ **支持 Llama3** - 与项目要求完全匹配
✅ **免费额度** - 新用户有免费credits
✅ **多模型支持** - 一个API访问几十种模型
✅ **OpenAI兼容** - 代码简单易用
✅ **注册简单** - Google账号即可

---

## 📋 配置步骤（5分钟搞定）

### 步骤 1: 注册并获取 API Key

1. **访问**: https://openrouter.ai/

2. **注册登录**:
   - 点击右上角 "Sign In"
   - 使用 Google 账号快速登录（推荐）
   - 或者用邮箱注册

3. **获取 API Key**:
   - 登录后访问: https://openrouter.ai/keys
   - 点击 "Create Key"
   - 给Key起个名字（如: legal-project）
   - 点击 "Create"
   - **立即复制API Key**（格式: `sk-or-v1-xxxxxxxxxx`）

4. **充值（可选）**:
   - 新用户通常有免费额度
   - 如果需要更多，可以充值 $5-10
   - 支持信用卡（但免费额度通常够开发用）

### 步骤 2: 安装依赖

```bash
# 安装 OpenAI SDK（OpenRouter兼容OpenAI接口）
pip install openai

# 如果已经安装过，确认版本 >= 1.0
pip install --upgrade openai
```

### 步骤 3: 设置 API Key

**方法 A: 设置环境变量（推荐）**

```bash
# Windows Git Bash
export OPENROUTER_API_KEY=sk-or-v1-你的实际API_Key

# Windows CMD
set OPENROUTER_API_KEY=sk-or-v1-你的实际API_Key

# Windows PowerShell
$env:OPENROUTER_API_KEY="sk-or-v1-你的实际API_Key"
```

**方法 B: 创建 .env 文件**

```bash
# 在项目根目录创建 .env 文件
echo OPENROUTER_API_KEY=sk-or-v1-你的实际API_Key > .env
```

### 步骤 4: 测试配置

```bash
python test_openrouter.py
```

**预期输出**:
```
============================================================
OpenRouter API Configuration Test
============================================================

OK: API Key found: sk-or-v1-xxx...

Test 1: Llama3-8B Model
SUCCESS!
Answer: A legal contract is a binding agreement...
Tokens: 45
Latency: 1.2s

ALL TESTS PASSED!
```

---

## 💻 在代码中使用

### 基本用法

```python
from utils.openrouter_llm import OpenRouterLLM

# 初始化
llm = OpenRouterLLM(model_name="llama3")

# 生成
result = llm.generate("What is a legal contract?")
print(result['text'])
print(f"Tokens: {result['tokens']}, Time: {result['latency']:.2f}s")
```

### 在 Agent 组件中使用

#### Query Analyzer
```python
# agent/query_analyzer/analyzer.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from utils.openrouter_llm import OpenRouterLLM

class QueryAnalyzer:
    def __init__(self):
        self.llm = OpenRouterLLM(model_name="llama3", max_tokens=200)

    def analyze(self, query: str):
        prompt = f"""Analyze this legal query:
Query: {query}

Classification: [factual/reasoning/comparative/procedural]
Key entities:
Required information:"""

        result = self.llm.generate(prompt)
        return result['text']
```

#### Verifier
```python
# agent/verifier/verifier.py
from utils.openrouter_llm import OpenRouterLLM

class Verifier:
    def __init__(self):
        self.llm = OpenRouterLLM(model_name="llama3", max_tokens=300)

    def verify(self, claim: str, evidence: str):
        prompt = f"""Verify if claim is supported by evidence:

Claim: {claim}
Evidence: {evidence}

Verdict: [SUPPORTED/CONTRADICTED/INSUFFICIENT]
Explanation:"""

        result = self.llm.generate(prompt)
        return result['text']
```

#### Reviser
```python
# agent/reviser/reviser.py
from utils.openrouter_llm import OpenRouterLLM

class Reviser:
    def __init__(self):
        self.llm = OpenRouterLLM(model_name="llama3", max_tokens=512)

    def revise(self, original: str, evidence: str, verification: str):
        prompt = f"""Revise the answer based on verification:

Original: {original}
Evidence: {evidence}
Verification: {verification}

Revised answer (grounded in evidence):"""

        result = self.llm.generate(prompt)
        return result['text']
```

---

## 📊 可用模型

| 模型名称 | OpenRouter ID | 速度 | 成本 | 推荐用途 |
|---------|--------------|------|------|---------|
| llama3 | meta-llama/llama-3-8b-instruct | ⚡⚡⚡⚡ | 💰 | 主要开发（推荐）|
| llama3-70b | meta-llama/llama-3-70b-instruct | ⚡⚡⚡ | 💰💰 | 复杂推理 |
| mixtral | mistralai/mixtral-8x7b-instruct | ⚡⚡⚡⚡ | 💰 | 长文本 |
| mistral | mistralai/mistral-7b-instruct | ⚡⚡⚡⚡⚡ | 💰 | 快速测试 |

---

## 💰 成本估算

### 免费额度
- 新用户通常有 $1-5 免费credits
- 足够完成开发阶段的测试

### Llama3-8B 价格
- 输入: ~$0.06 / 1M tokens
- 输出: ~$0.06 / 1M tokens
- **超级便宜！**

### 项目需求估算
```
开发阶段:
- Query Analyzer: ~200 tokens/query
- Verifier: ~300 tokens/verification
- Reviser: ~400 tokens/revision
- 总计: ~900 tokens/sample

测试100个样本 = 90,000 tokens
成本: ~$0.005 (约 ¥0.04)

完整开发 (1000次调用) ≈ $0.10 (约 ¥0.7)
```

**结论**: 几乎免费！免费额度绰绰有余！

---

## 🎯 优势对比

| 特性 | OpenRouter | Groq | 硅基流动 |
|-----|-----------|------|---------|
| 香港访问 | ✅ 稳定 | ❌ 受限 | ✅ 稳定 |
| 注册难度 | ⭐ 简单 | ⭐⭐⭐ 困难 | ⭐ 简单 |
| 免费额度 | ✅ 有 | ✅ 大量 | ✅ 有 |
| 支持Llama3 | ✅ 完美 | ✅ 完美 | ✅ 支持 |
| 模型选择 | ⭐⭐⭐⭐⭐ 几十种 | ⭐⭐⭐ 几种 | ⭐⭐⭐⭐ 多种 |
| API稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 推荐度 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |

**结论**: OpenRouter 最适合你的情况（香港+稳定+简单）

---

## ❓ 常见问题

### Q1: 需要信用卡吗？
A: 不一定。免费额度通常够用。需要更多时才需要充值。

### Q2: 速度快吗？
A: 很快！通常1-2秒响应。

### Q3: 支持哪些模型？
A: 几十种！包括 Llama3, Mistral, Claude, GPT等。查看: https://openrouter.ai/models

### Q4: 如果免费额度用完了？
A: 充值 $5-10 足够完成整个项目（实际可能只用 $1-2）

### Q5: 和 Groq 有什么区别？
A: OpenRouter 更稳定，支持更多模型，但速度可能稍慢一点（但仍然很快）

---

## 🔧 调试技巧

### 查看详细日志
```python
import logging
logging.basicConfig(level=logging.INFO)

from utils.openrouter_llm import OpenRouterLLM
llm = OpenRouterLLM(model_name="llama3")
```

### 测试不同模型
```python
# 测试更强大的模型
llm = OpenRouterLLM(model_name="llama3-70b")

# 测试更快的模型
llm = OpenRouterLLM(model_name="mistral")
```

### 错误处理
```python
try:
    result = llm.generate(prompt)
except Exception as e:
    print(f"API Error: {e}")
    # 处理错误
```

---

## 📝 下一步行动

1. ✅ 访问 https://openrouter.ai/ 注册
2. ✅ 获取 API Key
3. ✅ 设置环境变量
4. ✅ 运行 `python test_openrouter.py`
5. 📝 开始开发 Agent 组件
6. 🧪 测试你的代码
7. 🚀 准备完整实验

---

## 📞 需要帮助？

- **OpenRouter 文档**: https://openrouter.ai/docs
- **支持的模型**: https://openrouter.ai/models
- **定价**: https://openrouter.ai/docs#models

---

**配置完成后立即可以开始开发！比Groq更简单稳定！🎉**
