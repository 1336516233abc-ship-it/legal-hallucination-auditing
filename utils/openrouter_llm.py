"""
OpenRouter API 接口 - 支持多种开源模型
OpenRouter API Interface - Supports multiple open-source models

Author: Wang Zhengyao
Date: 2024-11-18
"""

import os
import time
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OpenRouterLLM:
    """OpenRouter API接口类 - 支持Llama3, Mistral等多种模型"""

    def __init__(
        self,
        model_name: str = "llama3",
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        初始化OpenRouter API接口

        Args:
            model_name: 模型名称 ("llama3", "llama3-70b", "mixtral", "mistral")
            api_key: OpenRouter API密钥
            max_tokens: 最大生成token数
            temperature: 温度参数 (0-1)
            top_p: nucleus sampling参数
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # 获取API key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY")

        if not self.api_key:
            raise ValueError(
                "未找到API Key! 请设置环境变量 OPENROUTER_API_KEY 或传入 api_key 参数"
            )

        # 初始化OpenAI兼容客户端
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
        except ImportError:
            raise ImportError("请安装OpenAI SDK: pip install openai")

        # 模型映射 - OpenRouter使用完整的模型路径
        self.model_map = {
            "llama3": "meta-llama/llama-3.1-8b-instruct",
            "llama3-8b": "meta-llama/llama-3.1-8b-instruct",
            "llama3.1": "meta-llama/llama-3.1-8b-instruct",
            "llama3-70b": "meta-llama/llama-3.1-70b-instruct",
            "mixtral": "mistralai/mixtral-8x7b-instruct",
            "mistral": "mistralai/mistral-7b-instruct",
            "mistral-7b": "mistralai/mistral-7b-instruct",
        }

        self.openrouter_model = self.model_map.get(
            model_name.lower(),
            "meta-llama/llama-3.1-8b-instruct"
        )

        # Token计数
        self.total_tokens = 0

        logger.info(f"OpenRouter API初始化成功: {self.openrouter_model}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict:
        """
        生成文本

        Args:
            prompt: 输入提示词
            max_tokens: 覆盖最大token数
            temperature: 覆盖温度参数
            stop_sequences: 停止序列列表

        Returns:
            包含'text', 'tokens', 'latency'的字典
        """
        start_time = time.time()

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.openrouter_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
                stop=stop_sequences,
            )

            generated_text = response.choices[0].message.content
            num_tokens = response.usage.total_tokens if response.usage else 0

            # 更新token计数
            self.total_tokens += num_tokens

            latency = time.time() - start_time

            return {
                "text": generated_text.strip() if generated_text else "",
                "tokens": num_tokens,
                "latency": latency,
            }

        except Exception as e:
            logger.error(f"OpenRouter API错误: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict:
        """
        聊天补全（支持多轮对话）

        Args:
            messages: 消息列表，每个消息包含'role'和'content'
            max_tokens: 覆盖最大token数
            temperature: 覆盖温度参数

        Returns:
            包含'text', 'tokens', 'latency'的字典
        """
        start_time = time.time()

        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.openrouter_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
            )

            generated_text = response.choices[0].message.content
            num_tokens = response.usage.total_tokens if response.usage else 0

            self.total_tokens += num_tokens

            latency = time.time() - start_time

            return {
                "text": generated_text.strip() if generated_text else "",
                "tokens": num_tokens,
                "latency": latency,
            }

        except Exception as e:
            logger.error(f"OpenRouter API错误: {e}")
            raise

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[Dict]:
        """
        批量生成

        Args:
            prompts: 提示词列表
            max_tokens: 覆盖最大token数
            temperature: 覆盖温度参数

        Returns:
            结果列表
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens, temperature)
            results.append(result)
        return results

    def get_token_count(self) -> int:
        """获取总token使用量"""
        return self.total_tokens

    def reset_token_count(self):
        """重置token计数器"""
        self.total_tokens = 0


# 便捷函数
def create_openrouter_llm(model_name: str = "llama3", **kwargs) -> OpenRouterLLM:
    """
    创建OpenRouter LLM实例的便捷函数

    Args:
        model_name: 模型名称
        **kwargs: 其他参数传递给OpenRouterLLM

    Returns:
        OpenRouterLLM实例
    """
    return OpenRouterLLM(model_name=model_name, **kwargs)


if __name__ == "__main__":
    # 简单测试
    print("测试OpenRouter API接口...")

    try:
        llm = OpenRouterLLM(model_name="llama3")
        result = llm.generate("What is a legal contract? Answer in one sentence.")

        print(f"✅ 成功!")
        print(f"回答: {result['text']}")
        print(f"Token: {result['tokens']}")
        print(f"耗时: {result['latency']:.2f}秒")

    except ValueError as e:
        print(f"❌ 错误: {e}")
        print("请设置 OPENROUTER_API_KEY 环境变量")
    except Exception as e:
        print(f"❌ 错误: {e}")
