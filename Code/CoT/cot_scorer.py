import os
import argparse
from typing import Optional
import concurrent.futures
import time

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

# Ensure project root and Code directory are on sys.path
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR = os.path.join(ROOT_DIR, 'Code')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from Code.Algo import PromptTemplate
from Code.getData7 import GetData
from cot_logger import CoTLogger
from cot_model_client import CoTModelClient

class CoTScorer:
    """
    使用思维链（Chain-of-Thought, CoT）提示进行作文评分：
    - 引导模型逐步依据量表进行分析，再给出最终分数
    - 仅输出最终 JSON（包含 score 与简要 justification），便于解析
    """
    def __init__(self, args: dict, scoring_range: tuple[int, int], essay_prompt: str, rubric: str):
        self.args = args
        self.score_range = (int(scoring_range[0]), int(scoring_range[1]))
        self.essay_prompt = essay_prompt
        self.rubric = rubric
        
        # 初始化日志记录器
        self.logger = CoTLogger(
            essay_set_id=args.get('essay_set'),
            model=args.get('model', 'unknown')
        )
        
        # 初始化独立的模型客户端
        self.model_client = CoTModelClient(logger=self.logger)

        output_format = (
            f"""Return JSON in this format:\n{{\n  \"score\": <integer between {self.score_range[0]} and {self.score_range[1]}>,\n  \"justification\": \"brief reasoning referencing rubric criteria\"\n}}\nNote: Return only the JSON object without any additional text."""
        )

        base_instruction = (
            """You are an experienced secondary school English teacher. \n"
            "Please evaluate the essay using a careful, step-by-step reasoning process:\n"
            "1) Identify key writing dimensions from the rubric\n"
            "2) Assess each dimension against the rubric levels\n"
            "3) Synthesize the evidence and resolve conflicts\n"
            "4) Determine the final score within the allowed range\n"
            "Think through the steps internally, then output ONLY the final JSON."""
        )

        self.template = PromptTemplate(
            base_instruction=base_instruction,
            scoring_criteria=rubric,
            output_format=output_format,
            args=self.args,
        )

    def score_essay(self, essay: str, essay_index: int = None) -> int:
        try:
            prompt = self.template.generate_prompt(essay, self.essay_prompt)
            response = self.model_client.chat(self.args['model'], prompt, temperature=self.args['temperature'])[0]
            
            # 导入独立的解析函数
            from cot_model_client import parse_json_response
            result = parse_json_response(response, min_score=self.score_range[0], logger=self.logger)

            if 'evaluations' in result and isinstance(result['evaluations'], list) and result['evaluations']:
                score = int(result['evaluations'][0].get('score', self.score_range[0]))
            else:
                score = int(result.get('score', self.score_range[0]))

            # clamp to allowed range
            score = max(min(score, self.score_range[1]), self.score_range[0])
            return score
        except Exception as e:
            error_msg = f"Error scoring essay: {e}"
            if essay_index is not None:
                self.logger.scoring_error(essay_index, str(e))
            else:
                self.logger.error(error_msg)
            return self.score_range[0]

    def evaluate(self, essays: list[str], human_scores: list[int]) -> dict:
        """并行评分多篇作文"""
        self.logger.scoring_start(len(essays))
        
        def score_single_essay(args):
            essay, idx = args
            try:
                score = self.score_essay(essay, essay_index=idx)
                self.logger.scoring_progress(idx + 1, len(essays), score)
                return score
            except Exception as e:
                self.logger.scoring_error(idx, str(e))
                return self.score_range[0]
        
        # 准备参数
        essay_args = [(essay, idx) for idx, essay in enumerate(essays)]
        
        # 并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(essays))) as executor:
            scores = list(tqdm(
                executor.map(score_single_essay, essay_args),
                total=len(essays),
                desc="CoT评分进度"
            ))
        
        # 统计结果
        successful = sum(1 for score in scores if score > self.score_range[0])
        failed = len(scores) - successful
        
        self.logger.scoring_complete(len(essays), successful, failed)
        
        return {
            'scores': scores,
            'human_scores': human_scores
        }


def read_test_set(input_path: str) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    # 兼容不同命名
    if 'human_score' in df.columns:
        score_col = 'human_score'
    elif 'domain1_score' in df.columns:
        score_col = 'domain1_score'
    else:
        raise ValueError("输入文件缺少 'human_score' 或 'domain1_score' 列")

    # 只保留必要列
    needed_cols = ['essay', score_col]
    if 'essay_set' in df.columns:
        needed_cols.append('essay_set')
    df = df[needed_cols].dropna(subset=['essay', score_col])
    df[score_col] = df[score_col].astype(int)
    return df.rename(columns={score_col: 'human_score'})


def run_cot_scoring(input_path: str, dataset: str, essay_set: Optional[int], model: str, temperature: float, out_dir: str):
    # Change to project root directory to ensure relative paths work correctly
    original_cwd = os.getcwd()
    os.chdir(ROOT_DIR)
    
    # 创建主日志记录器
    main_logger = CoTLogger(essay_set_id=essay_set, model=model)
    
    try:
        os.makedirs(out_dir, exist_ok=True)
        main_logger.info(f"正在读取测试集: {input_path}")
        df = read_test_set(input_path)
        main_logger.info(f"成功读取 {len(df)} 条测试数据")

        all_scores = []
        all_human = []
        all_rows = []

        # 如果有 essay_set 列则按组处理，否则需要传入 essay_set
        if 'essay_set' in df.columns:
            available_sets = sorted(df['essay_set'].unique())
            print(f"数据中包含的作文集合: {available_sets}")
            
            # 如果指定了特定的 essay_set，只处理该集合
            if essay_set is not None:
                if essay_set not in available_sets:
                    raise ValueError(f"指定的作文集合 {essay_set} 不存在于数据中。可用集合: {available_sets}")
                groups = [(essay_set, df[df['essay_set'] == essay_set])]
                print(f"处理指定的作文集合: {essay_set}")
            else:
                # 如果没有指定，处理所有集合
                groups = [(s, df[df['essay_set'] == s]) for s in available_sets]
                print(f"处理所有 {len(available_sets)} 个作文集合: {available_sets}")
        else:
            if essay_set is None:
                raise ValueError("输入文件不包含 'essay_set' 列，需通过 --essay_set 指定一个集合编号")
            groups = [(essay_set, df)]
            print(f"处理单个作文集合: {essay_set}")

        # 添加总体进度条
        print(f"\n开始处理 {len(groups)} 个作文集合...")
        overall_pbar = tqdm(groups, desc="总体进度", unit="集合", position=0, leave=True)
        
        for i, (s, g) in enumerate(overall_pbar, 1):
            overall_pbar.set_description(f"总体进度 - 当前: 作文集 {s}")
            print(f"\n=== 处理作文集合 {s} ({i}/{len(groups)}) ===")
            print(f"该集合包含 {len(g)} 篇作文")
            
            data = GetData(dataset, int(s))
            essay_prompt = data.get_prompt()
            score_range = data.get_score_range()
            rubric = data.get_rubric()
            print(f"已加载作文集合 {s} 的提示和评分标准")

            args = {
                'model': model,
                'temperature': temperature
            }
            scorer = CoTScorer(args, score_range, essay_prompt, rubric)

            print("开始并行评分...")
            
            # 使用并行评分方法
            essays = g['essay'].tolist()
            human_scores = g['human_score'].tolist()
            
            result = scorer.evaluate(essays, human_scores)
            g_scores = result['scores']
            
            all_scores.extend(g_scores)
            all_human.extend(g['human_score'].tolist())

            # 计算该集合的 kappa
            kappa = cohen_kappa_score(g['human_score'].tolist(), g_scores, weights='quadratic')
            print(f"作文集合 {s} 的加权 Cohen's Kappa: {kappa:.4f}")
            
            # 记录单个作文集的 kappa 值到日志
            main_logger.log_essay_set_kappa(s, kappa, len(g))

            # 保存该集合的结果
            result_df = pd.DataFrame({
                'essay_set': [s] * len(g),
                'essay': essays,
                'human_score': human_scores,
                'CoT_score': g_scores
            })
            result_file = os.path.join(out_dir, f'cot_results_set_{s}.xlsx')
            result_df.to_excel(result_file, index=False)
            print(f"结果已保存到: {result_file}")

            all_rows.extend(result_df.to_dict('records'))
            
            # 更新总体进度条信息
            overall_pbar.set_postfix({
                '当前集合': s,
                'Kappa': f"{kappa:.4f}",
                '已完成': f"{i}/{len(groups)}"
            })

        # 关闭总体进度条
        overall_pbar.close()
        
        # 聚合并计算整体 kappa
        main_logger.section("计算整体结果")
        overall_kappa = cohen_kappa_score(all_human, all_scores, weights='quadratic')
        main_logger.info(f"整体加权 Cohen's Kappa: {overall_kappa:.4f}")
        
        # 记录整体 kappa 值和关键参数
        main_logger.log_overall_kappa(overall_kappa, len(all_scores), model, temperature)
        
        # 记录分数分布统计
        main_logger.log_score_distribution(all_scores, all_human)

        # 保存汇总结果
        all_df = pd.DataFrame(all_rows)
        summary_file = os.path.join(out_dir, 'cot_results_summary.xlsx')
        all_df.to_excel(summary_file, index=False)
        main_logger.save_results(summary_file)



        main_logger.section("CoT 评分完成")
        main_logger.info(f"总共评分 {len(all_scores)} 篇作文")
        main_logger.info(f"整体 Kappa 值: {overall_kappa:.4f}")

        # 保存到输出目录
        scores_path = os.path.join(out_dir, 'cot_scores.xlsx')
        all_df.to_excel(scores_path, index=False)



        main_logger.info(f"CoT kappa: {overall_kappa:.4f}")
        main_logger.save_results(scores_path)
        
        # 关闭日志记录器
        main_logger.close()
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Chain-of-Thought scoring on test set and compute kappa')
    parser.add_argument('--input_path', default=os.path.join(ROOT_DIR, 'Data', 'Score(test set).xlsx'))
    parser.add_argument('--dataset', default='ASAP')
    parser.add_argument('--essay_set', default=1, type=int, help='指定要评分的作文集合编号(1-8)。如果不指定，将处理所有集合')
    parser.add_argument('--model', default='glm', choices=['glm', 'deepseek', 'qwen', 'doubao'])
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--out_dir', default=os.path.join('cot_results'))

    args = parser.parse_args()
    run_cot_scoring(
        input_path=args.input_path,
        dataset=args.dataset,
        essay_set=args.essay_set,
        model=args.model,
        temperature=args.temperature,
        out_dir=args.out_dir
    )