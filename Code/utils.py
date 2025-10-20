# -*- coding: utf-8 -*-
import os
import re
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional, Dict,Union
import requests
import json
import time
import config
import logging
from datetime import datetime
import threading
from typing import Optional
import hashlib
import ast
import argparse
from json_repair import repair_json
from openai import OpenAI
from volcenginesdkarkruntime import Ark
import inspect

# 检测是否在CoT环境中运行
def _is_cot_environment():
    """检测是否在CoT环境中运行"""
    frame = inspect.currentframe()
    try:
        while frame:
            filename = frame.f_code.co_filename
            if 'CoT' in filename or 'cot_' in filename:
                return True
            frame = frame.f_back
        return False
    finally:
        del frame



class ThreadSafeLogger:
    _instance = None
    _initialized = False

    def __new__(cls, essay_set_id: Optional[int] = None, disabled: bool = False):
        if cls._instance is None:
            cls._instance = super(ThreadSafeLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, essay_set_id: Optional[int] = None, disabled: bool = False):
        # 防止重复初始化
        if ThreadSafeLogger._initialized:
            return

        self.lock = threading.Lock() 
        self.disabled = disabled
        
        if disabled:
            # 禁用模式：不创建任何目录或文件
            self.essay_set_id = None
            self.base_dir = None
            self.timestamp = None
            self.log_dir = None
            self.normal_log_file = None
            self.error_log_file = None
        else:
            # 保存essay_set_id信息
            self.essay_set_id = essay_set_id
            # 基础日志目录
            self.base_dir = "logs"
            # 获取当前时间戳
            self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 如果没有提供 essay_set_id，使用默认值或不使用essay_set_id
            if essay_set_id is None:
                essay_set_id = None  # 保持为None，使用时间戳目录

            # 构建日志目录路径
            if essay_set_id:
                # 如果有 essay_set_id，创建带时间戳的子目录结构
                self.log_dir = os.path.join(
                    self.base_dir,
                    f'essay_set_{essay_set_id}',
                    self.timestamp
                )
            else:
                # 如果没有 essay_set_id，使用时间戳目录
                self.log_dir = os.path.join(
                    self.base_dir,
                    self.timestamp
                )

            # 创建目录
            os.makedirs(self.log_dir, exist_ok=True)

            # 创建日志文件路径
            self.normal_log_file = os.path.join(self.log_dir, "app.log")
            self.error_log_file = os.path.join(self.log_dir, "error.log")

            # 写入初始化信息到日志文件
            self._write_initialization_info()

        # 标记为已初始化
        ThreadSafeLogger._initialized = True

    def _write_initialization_info(self) -> None:
        """写入日志初始化信息"""
        if self.disabled:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        init_info = f"""
"""
        
        with self.lock:
            with open(self.normal_log_file, 'w', encoding='utf-8') as f:
                f.write(init_info)

    def _write_log(self, level: str, message: str) -> None:
        if self.disabled:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 添加essay_set信息到日志条目中
        essay_set_info = f"[Essay_Set_{self.essay_set_id}]" if self.essay_set_id is not None else "[Essay_Set_Unknown]"
        log_entry = f"[{timestamp}] {essay_set_info} [{level}] {message}\n"

        with self.lock:
            if level in ["ERROR", "WARNING"]:
                # 错误和警告日志写入错误日志文件
                with open(self.error_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
            else:
                # 普通日志写入普通日志文件
                with open(self.normal_log_file, 'a', encoding='utf-8') as f:
                    f.write(log_entry)

    def info(self, message: str) -> None:
        self._write_log("INFO", message)

    def warning(self, message: str) -> None:
        self._write_log("WARNING", message)

    def error(self, message: str) -> None:
        self._write_log("ERROR", message)

    def section(self, title: str) -> None:
        with self.lock:
            with open(self.normal_log_file, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                essay_set_info = f"[Essay_Set_{self.essay_set_id}]" if self.essay_set_id is not None else "[Essay_Set_Unknown]"
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"[{timestamp}] {essay_set_info} [SECTION] {title}\n")
                f.write("=" * 80 + "\n")

    def get_log_dir(self) -> str:
        return self.log_dir


logger = ThreadSafeLogger(disabled=_is_cot_environment())


class ResponseCache:
    def __init__(self, cache_dir="response_cache", disabled=False):
        self.disabled = disabled
        if disabled:
            self.cache_dir = None
        else:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        self.lock = threading.Lock()

    def get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[str]:
        """获取缓存的响应"""
        if self.disabled:
            return None
            
        key = self.get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)['response']
        return None

    def set(self, text: str, response: str):
        """保存响应到缓存"""
        if self.disabled:
            return
            
        key = self.get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")

        with self.lock:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'response': response}, f)

    def clear_cache(self):
        """清空所有缓存文件"""
        if self.disabled:
            return
            
        with self.lock:
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting cache file {file_path}: {e}")

# 根据环境决定是否禁用缓存和日志
_is_cot = _is_cot_environment()
_CACHE = ResponseCache(disabled=_is_cot)
class ZhipuAI(LLM):
    """智谱AI模型的Langchain封装"""

    model_name: str = "GLM-4-Flash"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 40000
    api_key: str = config.zhipu_key
    request_timeout: int = 180,
    @property
    def _llm_type(self) -> str:
        return "zhipu-ai"

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
                # 记录 token 使用信息
                prompt_tokens = response_json['usage']['prompt_tokens']
                completion_tokens = response_json['usage']['completion_tokens']
                total_tokens = response_json['usage']['total_tokens']

                # 创建详细的日志消息
                log_message = f"""
                        Token Usage Statistics:
                        - Prompt Tokens: {prompt_tokens}
                        - Completion Tokens: {completion_tokens}
                        - Total Tokens: {total_tokens}
                        - Max Tokens Limit: {self.max_tokens}
                        - Token Usage Rate: {(completion_tokens / self.max_tokens) * 100:.2f}%
                        """

                # with open('token_usage.log', 'a', encoding='utf-8') as f:
                #     f.write(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                #     f.write(log_message)
                #     f.write("-" * 50 + "\n")


                return content
            else:
                error_msg = f"API调用失败: {response.status_code} - {response.text}"
                # print(error_msg)
                # raise Exception(error_msg)

        except Exception as e:
            error_msg = f"""
                    Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
                    Error: {str(e)}
                    Prompt: {prompt}
                    """
            print(error_msg)
            # with open('error-log.txt', 'a', encoding='utf-8') as outf:
            #     outf.write(error_msg)
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

class DeepSeekAI(LLM):
    """DeepSeek AI模型的Langchain封装"""

    model_name: str = "deepseek-chat"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 8192  # 修改为DeepSeek支持的最大值
    api_key: str = config.deepseek_key
    request_timeout: int = 180

    @property
    def _llm_type(self) -> str:
        return "deepseek-ai"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """执行API调用"""
        
        try:
            # 确保max_tokens不超过DeepSeek的限制
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
            
            # 记录 token 使用信息
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # 创建详细的日志消息
            log_message = f"""
                    Token Usage Statistics:
                    - Prompt Tokens: {prompt_tokens}
                    - Completion Tokens: {completion_tokens}
                    - Total Tokens: {total_tokens}
                    - Max Tokens Limit: {adjusted_max_tokens}
                    - Token Usage Rate: {(completion_tokens / adjusted_max_tokens) * 100:.2f}%
                    """

            # with open('token_usage.log', 'a', encoding='utf-8') as f:
            #     f.write(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            #     f.write(log_message)
            #     f.write("-" * 50 + "\n")

            return content
            
        except Exception as e:
            error_msg = f"""
                    Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
                    Error: {str(e)}
                    Prompt: {prompt}
                    """
            print(error_msg)
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

class QwenAI(LLM):
    """阿里云Qwen大模型的Langchain封装"""

    # model_name: str = "qwen2-7b-instruct"# 默认使用qwen2-7b-instruct，可以根据需要更改
    model_name: str = "qwen2-1.5b-instruct"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 6144  # Qwen默认支持的最大值
    api_key: str = config.qwen_key
    request_timeout: int = 180

    @property
    def _llm_type(self) -> str:
        return "qwen-ai"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """执行API调用"""
        
        try:
            # 确保max_tokens不超过Qwen的限制
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
            
            # 记录 token 使用信息
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # 创建详细的日志消息
            log_message = f"""
                    Token Usage Statistics:
                    - Prompt Tokens: {prompt_tokens}
                    - Completion Tokens: {completion_tokens}
                    - Total Tokens: {total_tokens}
                    - Max Tokens Limit: {adjusted_max_tokens}
                    - Token Usage Rate: {(completion_tokens / adjusted_max_tokens) * 100:.2f}%
                    """

            # with open('token_usage.log', 'a', encoding='utf-8') as f:
            #     f.write(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            #     f.write(log_message)
            #     f.write("-" * 50 + "\n")

            return content
            
        except Exception as e:
            error_msg = f"""
                    Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
                    Error: {str(e)}
                    Prompt: {prompt}
                    """
            print(error_msg)
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

class DoubaoAI(LLM):
    """豆包大模型的Langchain封装"""

    model_name: str = "ep-20250419172906-4rv7j"
    temperature: float = 0.3
    top_p: float = 1.0
    max_tokens: int = 12288  # 豆包模型支持的最大token数
    api_key: str = config.doubao_key
    request_timeout: int = 180

    @property
    def _llm_type(self) -> str:
        return "doubao-ai"

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
            
            # 记录 token 使用信息
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            # 创建详细的日志消息
            log_message = f"""
                    Token Usage Statistics:
                    - Prompt Tokens: {prompt_tokens}
                    - Completion Tokens: {completion_tokens}
                    - Total Tokens: {total_tokens}
                    - Max Tokens Limit: {self.max_tokens}
                    - Token Usage Rate: {(completion_tokens / self.max_tokens) * 100:.2f}%
                    """

            # with open('token_usage.log', 'a', encoding='utf-8') as f:
            #     f.write(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            #     f.write(log_message)
            #     f.write("-" * 50 + "\n")

            return content
            
        except Exception as e:
            error_msg = f"""
                    Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
                    Error: {str(e)}
                    Prompt: {prompt}
                    """
            print(error_msg)
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

def try_parse_ast_to_json(function_string: str) -> tuple[str, dict]:
    """
     # 示例函数字符串
    function_string = "tool_call(first_int={'title': 'First Int', 'type': 'integer'}, second_int={'title': 'Second Int', 'type': 'integer'})"
    :return:
    """

    tree = ast.parse(str(function_string).strip())
    ast_info = ""
    json_result = {}
    # 查找函数调用节点并提取信息
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = node.func.id
            args = {kw.arg: kw.value for kw in node.keywords}
            ast_info += f"Function Name: {function_name}\r\n"
            for arg, value in args.items():
                ast_info += f"Argument Name: {arg}\n"
                ast_info += f"Argument Value: {ast.dump(value)}\n"
                json_result[arg] = ast.literal_eval(value)

    return ast_info, json_result

log = logging.getLogger(__name__)
def try_parse_json_object(input: str) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        log.info("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
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
    if input.startswith("```"):
        input = input[len("```"):]
    if input.startswith("```json"):
        input = input[len("```json"):]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        json_info = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:

            if len(json_info) < len(input):
                json_info, result = try_parse_ast_to_json(input)
            else:
                result = json.loads(json_info)

        except json.JSONDecodeError:
            log.exception("error loading json, json=%s", input)
            return json_info, {}
        else:
            if not isinstance(result, dict):
                log.exception("not expected dict type. type=%s:", type(result))
                return json_info, {}
            return json_info, result
    else:
        return input, result



def parse_json_response(response: str, max_retries: int = 2, min_score: int = 0) -> Dict:
    """
    解析LLM返回的JSON响应字符串，失败后最多重试2次
    Args:
        response: LLM返回的原始字符串
        max_retries: 最大重试次数，默认2次
        min_score: 评分标准的最小分值，由调用方根据当前作文集评分范围传入
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
            response = response.replace('‘', "'")
            response = response.replace('’', "'")
            response = response.replace('“', '"')
            response = response.replace('”', '"')

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
            logger.error(f"No 'scoring_criteria', 'rubric', or 'scoring rubric' found in the response: {response}")
            return {"score": min_score}

        except Exception as e:
            retry_count += 1
            logger.error(f"Parse error (attempt {retry_count}/{max_retries}): {str(e)}\nResponse: {response}")

            if retry_count >= max_retries:
                logger.error(f"Max retries ({max_retries}) reached, returning minimum score: {min_score}")
                return {"score": min_score}

            time.sleep(0.1)
            continue


def chat(model: str,
         prompt: str,
         temperature: float = 0.3,
         n: int = 1,
         top_p: float = 1,
         max_tokens: Optional[int] = None,
         timeout: int = 180,
         ) -> List[str]:
    logger = ThreadSafeLogger(disabled=_is_cot_environment())
    
    # 获取当前日志目录路径
    log_dir = logger.get_log_dir()
    
    # 在日志目录下创建conversation文件夹
    conversation_dir = os.path.join(log_dir, "conversation")
    os.makedirs(conversation_dir, exist_ok=True)
    
    # 生成唯一的文件名（时间戳+随机数+模型名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import random
    random_suffix = ''.join(random.choices('0123456789', k=4))
    filename = f"{timestamp}_{random_suffix}_{model.replace('/', '_')}.txt"
    filepath = os.path.join(conversation_dir, filename)
    
    # 获取调用函数的信息
    import inspect
    caller_frame = inspect.currentframe().f_back
    caller_info = ""
    if caller_frame:
        caller_function = caller_frame.f_code.co_name
        caller_file = os.path.basename(caller_frame.f_code.co_filename)
        caller_line = caller_frame.f_lineno
        caller_info = f"Called from: {caller_file}:{caller_function}:{caller_line}"
    
    # 使用全局缓存实例
    global _CACHE

    # 尝试从缓存获取响应
    cached_response = _CACHE.get(prompt)
    if cached_response:
        # 即使使用缓存，仍然记录对话
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"===== LLM CONVERSATION (CACHED) =====\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"{caller_info}\n\n")
            f.write("===== PROMPT =====\n\n")
            f.write(prompt)
            f.write("\n\n")
            f.write("===== RESPONSE (FROM CACHE) =====\n\n")
            f.write(cached_response)
            f.write("\n\n===== END OF CONVERSATION =====\n")
        
        return [cached_response]

    # 记录请求内容
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"===== LLM CONVERSATION =====\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"{caller_info}\n\n")
        f.write("===== PROMPT =====\n\n")
        f.write(prompt)
        f.write("\n\n")

    # 根据model选择使用哪种API和默认max_tokens
    if model.lower() in ['glm', 'zhipu', 'glm-4-flash']:
        # 智谱AI默认使用40000 max_tokens
        default_max_tokens = 40000
        llm = ZhipuAI(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or default_max_tokens,
            request_timeout=timeout
        )
    elif model.lower() in ['deepseek', 'deepseek-chat']:
        # DeepSeek默认使用8192 max_tokens
        default_max_tokens = 8192
        llm = DeepSeekAI(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or default_max_tokens,
            request_timeout=timeout
        )
    elif model.lower() in ['qwen', 'qwen-plus', 'qwen2-7b-instruct']:
        # Qwen默认使用4096 max_tokens
        default_max_tokens = 6144
        llm = QwenAI(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or default_max_tokens,
            request_timeout=timeout
        )
    elif model.lower() in ['Doubao','doubao', 'doubao-1.5-lite-32k']:
        default_max_tokens = 12288
        llm = DoubaoAI(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens or default_max_tokens,
            request_timeout=timeout
        )
    else:
        default_max_tokens = 40000
        logger.warning(f"未知模型 {model}，默认使用 ZhipuAI")
        llm = ZhipuAI(
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
                response = llm.invoke(prompt)
                _CACHE.set(prompt, response)  # 使用全局缓存实例存储响应
                
                # 记录响应内容
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write("===== RESPONSE =====\n\n")
                    f.write(response)
                    f.write("\n\n===== END OF CONVERSATION =====\n")
                
                responses.append(response)
            return responses

        except Exception as e:
            retries += 1
            # 记录错误信息
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"===== ERROR (Attempt {retries}/{max_retries}) =====\n\n")
                f.write(str(e))
                f.write("\n\n")
            
            time.sleep(3)
            if retries >= max_retries:
                # 记录最终失败
                with open(filepath, 'a', encoding='utf-8') as f:
                    f.write("===== FAILED AFTER MAX RETRIES =====\n")
                    f.write("\n\n===== END OF CONVERSATION =====\n")
                
                return [""]  # 返回空字符串而不是引发异常
    
    return [""]
