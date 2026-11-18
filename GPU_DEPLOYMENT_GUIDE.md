# GPU部署指南 - 显卡资源方案

## 📊 资源需求总结

### 模型资源需求
| 模型 | 显存需求(FP16) | 显存需求(INT8) | 显存需求(INT4) |
|------|---------------|---------------|---------------|
| Llama3-8B | 16GB | 8GB | 4GB |
| Mistral-7B | 14GB | 7GB | 3.5GB |
| Dense检索器 | 1-2GB | 1GB | - |
| **总计（单模型）** | **18GB** | **9GB** | **5GB** |

### 推理速度估算
| GPU型号 | 单次推理耗时 | 100样本耗时 | 全部实验耗时 |
|---------|-------------|------------|-------------|
| RTX 4090 | 1-2秒 | 3-5分钟 | 2-3小时 |
| RTX 3090 | 2-3秒 | 5-8分钟 | 3-5小时 |
| A100 | 0.5-1秒 | 2-3分钟 | 1-2小时 |
| T4 | 3-5秒 | 8-12分钟 | 5-8小时 |

---

## 方案对比

### 方案1: 本地GPU ⚡

#### 适用人群
- 已有高性能GPU（RTX 3090/4090, A5000+）
- 需要频繁运行实验
- 注重数据隐私

#### 硬件要求
- **最低配置**: RTX 3060 12GB（需使用INT8量化）
- **推荐配置**: RTX 4090 24GB / RTX 3090 24GB
- **豪华配置**: A100 40GB / H100 80GB

#### 优缺点
✅ 优点:
- 完全控制，随时运行
- 无API费用
- 数据完全本地

❌ 缺点:
- 需要硬件投资（¥5000+）
- 可能需要等待队列（共享环境）

#### 配置步骤
```bash
# 1. 安装CUDA和cuDNN
# 确保CUDA版本 >= 11.8

# 2. 安装PyTorch（GPU版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 验证GPU可用
python -c "import torch; print(torch.cuda.is_available())"

# 4. 修改 utils/llm_interface.py
# 使用本地模型加载，启用8bit量化
```

**代码示例**（本地GPU推理）:
```python
# utils/llm_interface.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMInterface:
    def __init__(self, model_name="llama3"):
        if model_name == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        # 8bit量化加载（节省显存）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",           # 自动分配设备
            load_in_8bit=True,           # 8bit量化
            torch_dtype=torch.float16,   # FP16精度
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def generate(self, prompt, max_tokens=512):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

### 方案2: 云服务器租用 ☁️ **（推荐）**

#### 为什么推荐？
- ✅ 成本可控（¥50-100完成整个项目）
- ✅ 高性能GPU（4090/A100）
- ✅ 按需使用，用完即停
- ✅ 预装环境，快速启动

#### 平台对比

##### 🇨🇳 国内平台（推荐）

**1. AutoDL** (https://www.autodl.com/)
- **价格**: RTX 4090 ¥2.8/小时，RTX 3090 ¥1.5/小时
- **优点**: 界面友好，稳定性好，网络快
- **缺点**: 高峰期可能无机器
- **推荐指数**: ⭐⭐⭐⭐⭐

**2. 恒源云** (https://gpushare.com/)
- **价格**: RTX 4090 ¥2.5/小时，RTX 3090 ¥1.2/小时
- **优点**: 价格便宜，机器充足
- **缺点**: 界面略复杂
- **推荐指数**: ⭐⭐⭐⭐

**3. 智星云** (https://www.ai-galaxy.cn/)
- **价格**: RTX 3090 ¥1.0/小时
- **优点**: 价格最低
- **缺点**: 稳定性一般
- **推荐指数**: ⭐⭐⭐

##### 🌍 国际平台

**4. Google Colab Pro** ($9.99/月)
- GPU: T4/V100（免费版有时间限制）
- 适合小规模实验

**5. AWS/Azure/GCP**
- 按小时计费，适合企业用户
- 价格较高（$1-3/小时）

#### 成本估算

**场景1: 顺利完成（理想情况）**
```
环境配置: 1小时 × ¥2.5 = ¥2.5
数据处理: 1小时 × ¥2.5 = ¥2.5
基线实验: 3小时 × ¥2.5 = ¥7.5
Agent实验: 4小时 × ¥2.5 = ¥10
消融实验: 3小时 × ¥2.5 = ¥7.5
调试优化: 2小时 × ¥2.5 = ¥5
-----------------------------------
总计: 14小时 = ¥35
```

**场景2: 需要调试（实际情况）**
```
额外调试时间: 10小时 × ¥2.5 = ¥25
总计: 24小时 = ¥60
```

**场景3: 多次重跑（保守估计）**
```
总计: 40小时 = ¥100
```

#### 使用步骤（以AutoDL为例）

**Step 1: 注册并充值**
```
1. 访问 https://www.autodl.com/
2. 注册账号
3. 充值 ¥100（推荐）
```

**Step 2: 创建实例**
```
1. 选择GPU: RTX 4090（24GB显存）
2. 选择镜像: PyTorch 2.0 + Python 3.10
3. 选择存储: 50GB系统盘 + 50GB数据盘
4. 点击"创建实例"
```

**Step 3: 连接实例**
```bash
# 方式1: Web终端（推荐新手）
直接在网页打开终端

# 方式2: SSH连接（推荐开发者）
ssh root@region-xx.autodl.com -p xxxxx

# 方式3: JupyterLab
打开提供的JupyterLab链接
```

**Step 4: 上传代码**
```bash
# 方式1: Git clone
git clone https://github.com/your-repo/legal-hallucination-auditing.git

# 方式2: 本地上传
# 使用AutoDL的文件管理器上传zip文件

# 方式3: 自动同步
# 配置AutoDL的文件同步功能
```

**Step 5: 安装依赖**
```bash
cd legal-hallucination-auditing
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
```

**Step 6: 运行实验**
```bash
# 测试GPU
python -c "import torch; print(torch.cuda.is_available())"

# 运行实验
python experiments/run_experiments.py --mode all
```

**Step 7: 下载结果**
```bash
# 打包结果
tar -czf results.tar.gz experiments/results/

# 下载到本地（通过Web界面或scp）
```

**Step 8: 关闭实例（重要！）**
```
实验完成后立即关机，避免浪费费用
```

#### 省钱技巧 💰

1. **使用低峰时段**: 晚上10点后机器充足，价格可能更低
2. **及时关机**: 实验完成立即关机，不要挂机
3. **批量运行**: 一次性运行所有实验，避免多次启动
4. **使用快照**: 环境配置好后创建快照，下次直接恢复
5. **选择合适GPU**:
   - 测试代码用T4/3060（便宜）
   - 正式实验用4090（快速）

---

### 方案3: API调用 🌐

#### 适用人群
- 没有GPU资源
- 追求最简单部署
- 样本量较小

#### 平台选择

##### 🚀 Groq (https://groq.com/) **强烈推荐**
- **价格**: 免费额度 + 超低价
- **速度**: 极快（500+ tokens/秒）
- **模型**: Llama3-8B, Mixtral-8x7B
- **优点**: 免费额度充足，速度快到离谱
- **推荐指数**: ⭐⭐⭐⭐⭐

**使用示例**:
```python
# utils/llm_interface.py
from groq import Groq

class LLMInterface:
    def __init__(self):
        self.client = Groq(api_key="your-api-key")

    def generate(self, prompt, max_tokens=512):
        response = self.client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
```

##### 💼 Together AI (https://together.ai/)
- **价格**: $0.2/百万tokens（Llama3-8B）
- **模型**: 大量开源模型
- **优点**: 稳定可靠

##### 🌏 国内API服务

**阿里云DashScope**
- 模型: 通义千问、Llama等
- 价格: 按tokens计费

**百度千帆**
- 模型: 文心一言、Llama2等
- 价格: 按tokens计费

#### 成本估算

**使用Groq (免费额度)**
```
免费额度: 每天7000次请求
本项目需求: 约1500次请求
成本: ¥0（免费）
```

**使用Together AI**
```
总token数: 2.25M tokens
成本: $0.45（约¥3.2）
```

#### 配置步骤

**Step 1: 获取API Key**
```
1. 访问 https://groq.com/
2. 注册账号
3. 获取API Key
```

**Step 2: 修改代码**
```python
# utils/llm_interface.py
import os
from groq import Groq

class LLMInterface:
    def __init__(self, model_name="llama3", api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)

        self.model_map = {
            "llama3": "llama3-8b-8192",
            "mixtral": "mixtral-8x7b-32768"
        }
        self.model = self.model_map.get(model_name, "llama3-8b-8192")

    def generate(self, prompt, max_tokens=512, temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return None
```

**Step 3: 设置环境变量**
```bash
# Windows
set GROQ_API_KEY=your-api-key-here

# Linux/Mac
export GROQ_API_KEY=your-api-key-here

# 或创建 .env 文件
echo "GROQ_API_KEY=your-api-key" > .env
```

**Step 4: 运行实验**
```bash
python experiments/run_experiments.py --mode all --use_api
```

---

## 🎯 针对你们团队的推荐

### 推荐方案组合

#### 阶段1: 开发测试（第1-3天）
**使用Groq API**
- 成本: ¥0（免费）
- 用途: 代码调试、小样本测试
- 优点: 快速迭代，无需GPU

#### 阶段2: 正式实验（第4-5天）
**租用AutoDL 4090**
- 成本: ¥40-60
- 用途: 运行完整实验
- 优点: 可控性好，速度快

#### 阶段3: 补充实验（第6-7天）
**继续使用API或保留GPU**
- 根据需要灵活选择

### 总成本估算
```
Groq API (测试): ¥0
AutoDL GPU (实验): ¥50
应急预算: ¥30
-------------------
总计: ¥80
```

---

## 📝 实施建议

### 第1步: 立即行动（今天）
1. **Yin Fengling**: 在本地笔记本配置Groq API，测试数据处理
2. **Qin Mengmeng**: 使用API测试检索系统
3. **Wang Zhengyao**: 使用API测试Agent组件
4. **Li Xinru**: 准备AutoDL账号并充值¥100

### 第2步: 切换到GPU（11月21日）
1. **Li Xinru**创建AutoDL实例
2. 上传代码和数据
3. 运行完整实验套件

### 第3步: 成本控制
1. 代码调试在本地完成（使用API）
2. GPU只用于正式实验
3. 实验完成立即关机

---

## ❓ 常见问题

### Q1: 如果没有预算怎么办？
**答**: 完全使用Groq免费API，速度虽慢但可完成

### Q2: 如果实验运行时间超出预期？
**答**:
1. 减少样本量（从512降到100）
2. 只运行关键实验
3. 使用更快的GPU（A100）

### Q3: 多人如何共享GPU？
**答**:
1. 使用Jupyter Hub多用户模式
2. 或者分时段使用（排班）
3. 或者每人租用自己的实例

### Q4: 如何确保GPU不浪费？
**答**:
1. 使用自动关机脚本
2. 设置使用时长提醒
3. 监控GPU利用率

---

## 🔧 优化技巧

### 提升推理速度
```python
# 1. 使用FP16精度
model = model.half()

# 2. 启用Flash Attention
# pip install flash-attn
model.config.use_flash_attention = True

# 3. 批量推理
def batch_inference(prompts, batch_size=4):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        outputs = model.generate(batch)
        results.extend(outputs)
    return results
```

### 减少显存占用
```python
# 1. 使用INT8量化
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# 2. 使用INT4量化（更激进）
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# 3. 清理缓存
import torch
torch.cuda.empty_cache()

# 4. 使用gradient checkpointing（训练时）
model.gradient_checkpointing_enable()
```

---

## 📊 方案总结对比

| 方案 | 成本 | 速度 | 难度 | 推荐度 |
|------|------|------|------|--------|
| 本地GPU | ¥0（已有）或¥5000+ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 云服务器 | ¥50-100 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Groq API | ¥0 | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| 其他API | ¥3-10 | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |

---

## 🎓 学习资源

- **模型量化**: https://huggingface.co/docs/transformers/quantization
- **AutoDL教程**: https://www.autodl.com/docs
- **Groq文档**: https://console.groq.com/docs
- **GPU性能对比**: https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/

---

**最终建议**: 先用Groq API测试（免费），正式实验租AutoDL的4090（¥50-60），总成本¥50-60即可完成整个项目！

**预祝实验成功！🚀**
