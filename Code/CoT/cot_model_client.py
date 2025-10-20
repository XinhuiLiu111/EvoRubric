# -*- coding: utf-8 -*-
"""
CoT 评分专用的模型调用客户端
独立于 Code/utils.py，使用 CoTLogger 进行日志记录
"""

import os
import sys
import time
import json
import hashlib
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
from openai import OpenAI
from volcenginesdkarkruntime import Ark

# 添加 Code 目录到路径以导入配置
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT_DIR, 'Code')
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import config
import re
from json_repair import repair_json


class CoTResponseCache:
    """CoT 专用的响应缓存系统（禁用模式 - 不创建任何文件或目录）"""
    
    def __init__(self, cache_dir=None):
        # 禁用缓存功能 - 不创建任何目录或文件
        self.cache_dir = None
        self.lock = threading.Lock()
        self._disabled = True  # 标记缓存已禁用

    def get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[str]:
        """获取缓存的响应（禁用模式直接返回None）"""
        if self._disabled:
            return None
        
        key = self.get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)['response']
        return None

    def set(self, text: str, response: str):
        """保存响应到缓存（禁用模式不执行任何操作）"""
        if self._disabled:
            return
        
        key = self.get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        with self.lock:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'response': response}, f)


class CoTZhipuAI(LLM):
    """智谱AI模型的Langchain封装 - CoT专用版本"""

    model_name: str = "GLM-4-Flash"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 40000
    api_key: str = config.zhipu_key
    request_timeout: int = 180

    @property
    def _llm_type(self) -> str:
        return "cot-zhipu-ai"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """执行API调用"""
        messages = [{
            "role": "user",
            "content": prompt
        }]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "messages": messages,
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(
                'https://open.bigmodel.cn/api/paas/v4/chat/completions',
                headers=headers,
                json=payload,
                timeout=self.request_timeout
            )

            if response.status_code == 200:
                response_json = response.json()
                content = response_json['choices'][0]['message']['content']
                return content
            else:
                raise Exception(f"API调用失败: {response.status_code} - {response.text}")

        except Exception as e:
            raise e

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }


class CoTDeepSeekAI(LLM):
    """DeepSeek AI模型的Langchain封装 - CoT专用版本"""

    model_name: str = "deepseek-chat"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 8192
    api_key: str = config.deepseek_key
    request_timeout: int = 180

    @property
    def _llm_type(self) -> str:
        return "cot-deepseek-ai"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """执行API调用"""
        try:
            adjusted_max_tokens = min(self.max_tokens, 8192)
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=adjusted_max_tokens,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            return content
            
        except Exception as e:
            raise e

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }


class CoTQwenAI(LLM):
    """阿里云Qwen大模型的Langchain封装 - CoT专用版本"""

    model_name: str = "qwen2-1.5b-instruct"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 6144
    api_key: str = config.qwen_key
    request_timeout: int = 180

    @property
    def _llm_type(self) -> str:
        return "cot-qwen-ai"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """执行API调用"""
        try:
            adjusted_max_tokens = min(self.max_tokens, 6144)
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=adjusted_max_tokens,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            return content
            
        except Exception as e:
            raise e

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }


class CoTDoubaoAI(LLM):
    """豆包大模型的Langchain封装 - CoT专用版本"""

    model_name: str = "ep-20250419172906-4rv7j"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 12288
    api_key: str = config.doubao_key
    request_timeout: int = 180

    @property
    def _llm_type(self) -> str:
        return "cot-doubao-ai"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """执行API调用"""
        try:
            client = Ark(api_key=self.api_key,
                         base_url="https://ark.cn-beijing.volces.com/api/v3")

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            return content
            
        except Exception as e:
            raise e

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }


class CoTModelClient:
    """CoT 评分专用的模型客户端"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.cache = CoTResponseCache()
    
    def chat(self, 
             model: str,
             prompt: str,
             temperature: float = 0.3,
             n: int = 1,
             top_p: float = 1,
             max_tokens: Optional[int] = None,
             timeout: int = 180) -> List[str]:
        """
        CoT 专用的聊天函数，使用 CoTLogger 记录对话
        """
        
        # 尝试从缓存获取响应
        cached_response = self.cache.get(prompt)
        if cached_response:
            if self.logger:
                self.logger.info(f"使用缓存响应 - 模型: {model}")
            return [cached_response]

        # 根据model选择使用哪种API和默认max_tokens
        if model.lower() in ['glm', 'zhipu', 'glm-4-flash']:
            default_max_tokens = 40000
            llm = CoTZhipuAI(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or default_max_tokens,
                request_timeout=timeout
            )
        elif model.lower() in ['deepseek', 'deepseek-chat']:
            default_max_tokens = 8192
            llm = CoTDeepSeekAI(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or default_max_tokens,
                request_timeout=timeout
            )
        elif model.lower() in ['qwen', 'qwen-plus', 'qwen2-7b-instruct']:
            default_max_tokens = 6144
            llm = CoTQwenAI(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or default_max_tokens,
                request_timeout=timeout
            )
        elif model.lower() in ['doubao', 'doubao-1.5-lite-32k']:
            default_max_tokens = 12288
            llm = CoTDoubaoAI(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or default_max_tokens,
                request_timeout=timeout
            )
        else:
            default_max_tokens = 40000
            if self.logger:
                self.logger.warning(f"未知模型 {model}，默认使用 ZhipuAI")
            llm = CoTZhipuAI(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens or default_max_tokens,
                request_timeout=timeout
            )

        retries = 0
        max_retries = 10

        while retries < max_retries:
            try:
                responses = []
                for _ in range(n):
                    if self.logger:
                        self.logger.info(f"调用模型 {model} - 温度: {temperature}")
                    
                    response = llm.invoke(prompt)
                    self.cache.set(prompt, response)
                    
                    if self.logger:
                        self.logger.info(f"模型响应成功 - 长度: {len(response)} 字符")
                    
                    responses.append(response)
                return responses

            except Exception as e:
                retries += 1
                if self.logger:
                    self.logger.error(f"模型调用失败 (尝试 {retries}/{max_retries}): {str(e)}")
                
                time.sleep(3)
                if retries >= max_retries:
                    if self.logger:
                        self.logger.error(f"模型调用最终失败，已达到最大重试次数")
                    return [""]


def try_parse_json_object(input_str: str) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    result = None
    try:
        # Try parse first
        result = json.loads(input_str)
    except json.JSONDecodeError:
        pass

    if result:
        return input_str, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input_str)
    input_str = "{" + _match.group(1) + "}" if _match else input_str

    # Clean up json string.
    input_str = (
        input_str.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input_str.startswith("```"):
        input_str = input_str[len("```"):]
    if input_str.startswith("```json"):
        input_str = input_str[len("```json"):]
    if input_str.endswith("```"):
        input_str = input_str[: len(input_str) - len("```")]

    try:
        result = json.loads(input_str)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        json_info = str(repair_json(json_str=input_str, return_objects=False))
        try:
            result = json.loads(json_info)
        except json.JSONDecodeError:
            result = {}

    return input_str, result


def parse_json_response(response: str, max_retries: int = 2, min_score: int = 0, logger=None) -> Dict:
    """
    解析LLM返回的JSON响应字符串，失败后最多重试2次
    Args:
        response: LLM返回的原始字符串
        max_retries: 最大重试次数，默认2次
        min_score: 评分标准的最小分值，由调用方根据当前作文集评分范围传入
        logger: 日志记录器
    Returns:
        解析后的字典对象:
        1. 包含evaluations时返回完整的评分及解释
        2. 只有score时返回分数
        3. 包含scoring_criteria/rubric时返回评分标准
        4. 全部失败则返回评分标准的最小分值
    """
    retry_count = 0

    while True:
        try:
            if not response or not isinstance(response, str):
                return {"score": min_score}

            # 1. 预处理响应文本
            response = response.strip()
            response = response.replace(''', "'")
            response = response.replace(''', "'")
            response = response.replace('"', '"')
            response = response.replace('"', '"')

            # 移除所有可能的markdown标记和多余空白
            response = response.replace('```json', '').replace('```', '')
            response = ' '.join(response.split())  # 规范化空白字符

            # 2. 解析JSON
            try:
                # 首先尝试标准JSON解析
                parsed = json.loads(response)
            except json.JSONDecodeError:
                # 使用官方的JSON修复工具
                _, parsed = try_parse_json_object(response)
                if not parsed:
                    score_match = re.search(r'"score"\s*:\s*(\d+)', response)
                    if score_match:
                        return {"score": max(min_score, int(score_match.group(1)))}
                    return {"score": min_score}

            # 3. 处理解析结果
            # 情况1：包含evaluations - 完整的评分及解释
            if "evaluations" in parsed:
                for eval_item in parsed["evaluations"]:
                    # 先检查evaluation本身的score
                    if "score" in eval_item:
                        eval_item["score"] = max(min_score, int(eval_item["score"]))
                        return parsed
                    
                    # 如果evaluation没有score，再检查criteria_analysis中的score
                    if "criteria_analysis" in eval_item:
                        for criteria in eval_item["criteria_analysis"]:
                            if "score" in criteria:
                                eval_item["score"] = max(min_score, int(criteria["score"]))
                                return parsed
                
                # 如果evaluations中没找到score
                if logger:
                    logger.error(f"No score found in evaluations: {response}")
                return {"score": min_score}
            
            # 情况2：只包含score - 返回分数
            elif "score" in parsed:
                return {"score": max(min_score, int(parsed["score"]))}
            
            # 情况3：包含评分标准
            elif any(key in parsed for key in ["scoring_criteria", "rubric", "scoring rubric"]):
                return {"scoring_criteria": parsed.get("scoring_criteria") or
                                            parsed.get("rubric") or
                                            parsed.get("scoring rubric")}
            #如果"scoring_criteria", "rubric", "scoring rubric"都不存在，则使用logger.error记录错误
            if logger:
                logger.error(f"No 'scoring_criteria', 'rubric', or 'scoring rubric' found in the response: {response}")
            return {"score": min_score}

        except Exception as e:
            retry_count += 1
            if logger:
                logger.error(f"Parse error (attempt {retry_count}/{max_retries}): {str(e)}\nResponse: {response}")

            if retry_count >= max_retries:
                if logger:
                    logger.error(f"Max retries ({max_retries}) reached, returning minimum score: {min_score}")
                return {"score": min_score}

            time.sleep(0.1)
            continue  # 返回空字符串而不是引发异常
        
        return [""]