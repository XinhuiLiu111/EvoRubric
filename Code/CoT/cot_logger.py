# -*- coding: utf-8 -*-
"""
CoT评分专用的简化日志系统
专门为Chain-of-Thought评分过程设计，记录评分相关的关键信息
"""

import os
import logging
from datetime import datetime
from typing import Optional


class CoTLogger:
    """CoT评分专用的简化日志记录器"""
    
    def __init__(self, essay_set_id: Optional[int] = None, model: str = "unknown"):
        """
        初始化CoT日志记录器
        
        Args:
            essay_set_id: 作文集ID
            model: 使用的模型名称
        """
        self.essay_set_id = essay_set_id
        self.model = model
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建日志目录和文件
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'CoT')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 日志文件路径
        log_filename = f"{model}_cot_scoring_{self.timestamp}.log"
        if essay_set_id:
            log_filename = f"{model}_cot_set{essay_set_id}_{self.timestamp}.log"
        
        self.log_file = os.path.join(self.log_dir, log_filename)
        self.error_file = os.path.join(self.log_dir, f"{model}_cot_errors_{self.timestamp}.log")
        
        # 配置日志格式（仅用于内存中的日志记录器）
        self.log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
        self.date_format = "%Y-%m-%d %H:%M:%S"
        
        # 创建内存日志记录器（不写入文件）
        self._setup_loggers()
        
        # 记录开始信息（仅在内存中）
        self.info(f"CoT评分开始 - 作文集: {essay_set_id}, 模型: {model}")
    
    def _setup_loggers(self):
        """设置日志记录器"""
        # 主日志记录器
        self.logger = logging.getLogger(f"cot_scorer_{self.timestamp}")
        self.logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(self.log_format, self.date_format)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 错误日志记录器
        self.error_logger = logging.getLogger(f"cot_error_{self.timestamp}")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.handlers.clear()
        
        # 创建错误文件处理器
        error_file_handler = logging.FileHandler(self.error_file, encoding='utf-8')
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)
        
        # 添加错误处理器
        self.error_logger.addHandler(error_file_handler)
    
    def info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误日志"""
        self.logger.error(message)
        self.error_logger.error(message)
    
    def section(self, title: str):
        """记录分段标题"""
        separator = "=" * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"{title}")
        self.logger.info(separator)
    
    def scoring_start(self, total_essays: int):
        """记录评分开始"""
        self.section("开始CoT评分")
        self.info(f"总作文数量: {total_essays}")
        self.info(f"使用模型: {self.model}")
        self.info(f"作文集: {self.essay_set_id}")
    
    def scoring_progress(self, current: int, total: int, score: int, essay_id: str = None):
        """记录评分进度"""
        progress = (current / total) * 100
        essay_info = f" (ID: {essay_id})" if essay_id else ""
        self.info(f"评分进度: {current}/{total} ({progress:.1f}%) - 分数: {score}{essay_info}")
    
    def scoring_error(self, essay_index: int, error_msg: str, essay_id: str = None):
        """记录评分错误"""
        essay_info = f" (ID: {essay_id})" if essay_id else ""
        self.error(f"作文 {essay_index}{essay_info} 评分失败: {error_msg}")
    
    def scoring_complete(self, total_essays: int, successful: int, failed: int, kappa: float = None):
        """记录评分完成"""
        self.section("CoT评分完成")
        self.info(f"总作文数: {total_essays}")
        self.info(f"成功评分: {successful}")
        self.info(f"失败评分: {failed}")
        self.info(f"成功率: {(successful/total_essays)*100:.1f}%")
        if kappa is not None:
            self.info(f"Kappa值: {kappa:.4f}")
    
    def log_essay_set_kappa(self, essay_set: int, kappa: float, total_essays: int):
        """记录单个作文集的 kappa 值"""
        self.info(f"作文集 {essay_set} - Kappa值: {kappa:.4f} (共{total_essays}篇)")
    
    def log_overall_kappa(self, kappa: float, total_essays: int, model: str, temperature: float):
        """记录整体 kappa 值和关键参数"""
        self.section("整体评分结果")
        self.info(f"整体 Kappa 值: {kappa:.4f}")
        self.info(f"总评分作文数: {total_essays}")
        self.info(f"使用模型: {model}")
        self.info(f"温度参数: {temperature}")
    
    def log_kappa_comparison(self, current_kappa: float, baseline_kappa: float = None):
        """记录 kappa 值对比"""
        if baseline_kappa is not None:
            improvement = ((current_kappa - baseline_kappa) / baseline_kappa * 100) if baseline_kappa > 0 else 0
            self.info(f"Kappa 改进: {improvement:+.2f}% (当前: {current_kappa:.4f}, 基线: {baseline_kappa:.4f})")
        else:
            self.info(f"当前 Kappa 值: {current_kappa:.4f}")
    
    def log_score_distribution(self, scores: list, human_scores: list):
        """记录分数分布统计"""
        import numpy as np
        self.section("分数分布统计")
        self.info(f"AI评分 - 均值: {np.mean(scores):.2f}, 标准差: {np.std(scores):.2f}, 中位数: {np.median(scores):.2f}")
        self.info(f"人工评分 - 均值: {np.mean(human_scores):.2f}, 标准差: {np.std(human_scores):.2f}, 中位数: {np.median(human_scores):.2f}")
        
        # 记录分数范围分布
        ai_min, ai_max = min(scores), max(scores)
        human_min, human_max = min(human_scores), max(human_scores)
        self.info(f"AI评分范围: {ai_min}-{ai_max}, 人工评分范围: {human_min}-{human_max}")
    
    def save_results(self, file_path: str):
        """记录结果保存"""
        self.info(f"评分结果已保存至: {file_path}")
    
    def get_log_dir(self) -> str:
        """获取日志目录路径"""
        return self.log_dir
    
    def close(self):
        """关闭日志记录器"""
        self.info("CoT评分日志记录结束")
        
        # 关闭所有处理器
        for handler in self.logger.handlers:
            handler.close()
        for handler in self.error_logger.handlers:
            handler.close()