# scoring_with_best_template.py
# 使用最佳评分标准对测试集进行评分并计算性能指标

import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import time
from typing import List, Dict, Tuple, Any, Optional
import argparse
import matplotlib
from utils import _CACHE
matplotlib.rcParams['font.family'] = ['SimHei']  # 使用中文黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt
# 导入现有代码中的工具函数
from utils import chat, parse_json_response, ThreadSafeLogger

class BestTemplateScorer:
    """使用最佳评分标准的评分系统"""
    
    def __init__(self, 
                 model: str, 
                 temperature: float, 
                 scoring_range: Tuple[int, int],
                 essay_prompt: str, 
                 best_template_path: str,
                 essay_set_id: int,
                 out_dir: str,
                 max_workers: int = 8):
        """
        初始化评分系统
        
        Args:
            model: 使用的LLM模型名称
            temperature: 模型温度参数
            scoring_range: 评分范围元组 (min_score, max_score)
            essay_prompt: 作文提示
            best_template_path: 最佳评分标准文件路径
            essay_set_id: 作文集ID，用于从模板文件中选择对应的评分标准
            out_dir: 输出目录
            max_workers: 并行工作线程数
        """
        self.model = model
        self.temperature = temperature
        self.score_range = scoring_range
        self.essay_prompt = essay_prompt
        self.out_dir = out_dir
        self.essay_set_id = essay_set_id
        self.max_workers = min(max_workers, (os.cpu_count() or 1) * 2)
        self.logger = ThreadSafeLogger(essay_set_id=self.essay_set_id)
        
        # 加载最佳评分标准
        self.template = self._load_template(best_template_path)
        
    def _load_template(self, template_path: str):
        """加载最佳评分标准模板"""
        # 读取Excel文件
        df = pd.read_excel(template_path)
        
        # 确保essay_set列为整数类型
        if 'essay_set' in df.columns:
            df['essay_set'] = df['essay_set'].astype(int)
        
        # 根据essay_set筛选出对应的评分标准
        template_row = df[df['essay_set'] == int(self.essay_set_id)]
        
        if template_row.empty:
            raise ValueError(f"未找到essay_set_{self.essay_set_id}的最佳评分标准")
        
        # 获取评分标准内容
        scoring_criteria = template_row['最佳评分标准(doubao)'].values[0]
        
        # 创建PromptTemplate对象
        output_format = f'''
        IMPORTANT JSON FORMATTING RULES:
        1. Ensure all properties within objects are separated by commas
        2. Do not add a comma after the last property in an object
        3. All string values must be properly quoted
        Output your evaluation as a valid JSON response using this exact structure:
                {{
                    "evaluations": [
                        {{
                            "essay_number": 1,
                            "criteria_analysis": [
                                {{
                                    "criterion": "evaluation dimension (e.g., content quality, organization, language use, or other relevant aspect)",
                                    "explanation": "Describe how the essay demonstrates this criterion.Focus on describing the evidence rather than quoting it directly",
                                    "criterion_level": <integer between {self.score_range[0]} and {self.score_range[1]}>
                                }},
                                // Add more criteria as needed
                            ],
                            "justification": "Explain how criteria led to final score",
                            "score": <integer between {self.score_range[0]} and {self.score_range[1]}>
                        }}
                    ]
                }}'''

        base_instruction = '''You are an experienced secondary school English teacher. Please evaluate the following essay following this systematic process:

                Step 1: Criterion-Based Assessment
                For each criterion in the scoring rubric, you must:
                - Quote at least 2 specific text evidence that relates to this criterion
                - Explain in detail how the evidence demonstrates the criterion
                - Determine the score level at which this criterion is met
                - Identify any language or structural elements that support your evaluation

                Step 2: Consider the overall response
                - Review all criteria evaluations together
                - Determine if the evidence consistently supports a particular score level
                - Identify and resolve any conflicts between criteria scores
                - Consider how different criteria interact and influence each other

                Step 3: Provide final score 
                - justification that references the specific criteria
                - provide score
                '''

        return {
            "base_instruction": base_instruction,
            "scoring_criteria": scoring_criteria,
            "output_format": output_format
        }
    
    def generate_prompt(self, essay: str) -> str:
        """生成评分prompt"""
        return f"""
        {self.template["base_instruction"]}

        Writing Task:
        {self.essay_prompt}

        Essay to evaluate:
        {essay}

        Scoring Criteria:
        {self.template["scoring_criteria"]}

        {self.template["output_format"]}
        """
    
    def _score_single_essay(self, essay: str) -> Tuple[int, Dict]:
        """评分单篇作文"""
        try:
            # 生成输入prompt
            prompt = self.generate_prompt(essay)

            # 保存生成的prompt和响应到同一个文件
            conversation_dir = os.path.join(self.out_dir, 'conversations')
            os.makedirs(conversation_dir, exist_ok=True)

            # 生成一个唯一的文件名
            import hashlib
            essay_hash = hashlib.md5(essay.encode('utf-8')).hexdigest()[:8]
            conversation_filename = f"conversation_{essay_hash}.txt"
            conversation_path = os.path.join(conversation_dir, conversation_filename)

            # 先保存输入prompt
            with open(conversation_path, 'w', encoding='utf-8') as f:
                f.write("=== 输入 Prompt ===\n")
                f.write(prompt)
                f.write("\n\n")

            # 调用API获取响应
            response = chat(self.model, prompt, temperature=self.temperature)[0]

            # 追加保存输出response
            with open(conversation_path, 'a', encoding='utf-8') as f:
                f.write("=== 模型响应 ===\n")
                f.write(response)

            # 使用parse_json_response解析结果进行评分，但不再保存到文件
            parsed_result = parse_json_response(response, min_score=self.score_range[0])

            if 'evaluations' in parsed_result:
                eval_data = parsed_result['evaluations'][0]
                score = eval_data['score']
                explanation = {
                    'criteria_analysis': eval_data.get('criteria_analysis', []),
                    'justification': eval_data.get('justification', '')
                }
            else:
                score = parsed_result.get('score', self.score_range[0])
                explanation = {}

            # 确保分数在有效范围内
            score = max(min(score, self.score_range[1]), self.score_range[0])

            # print(f"评分结果: {score}")
            return score, explanation

        except Exception as e:
            self.logger.error(f"Error scoring essay: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())  # 记录完整堆栈跟踪
            return self.score_range[0], {}

    def evaluate(self, essays: List[str], human_scores: List[int]) -> Dict[str, Any]:
        """评估多篇作文并计算性能指标"""
        self.logger.section("Best Template Evaluation")
        self.logger.info(
            f"配置信息:\n"
            f"- 模型: {self.model}\n"
            f"- 温度: {self.temperature}\n"
            f"- 评分范围: {self.score_range}\n"
            f"- 并行线程数: {self.max_workers}"
        )
        
        # 预分配列表，确保顺序
        scores = [self.score_range[0]] * len(essays)
        evaluations = [{}] * len(essays)
        
        # 使用线程池并行评分
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._score_single_essay, essay): idx
                for idx, essay in enumerate(essays)
            }
            
            # 使用tqdm创建进度条
            with tqdm(total=len(essays), desc="评分进度") as pbar:
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        score, evaluation = future.result()
                        scores[idx] = score
                        evaluations[idx] = evaluation
                    except Exception as e:
                        self.logger.error(f"评分第{idx}篇作文失败: {str(e)}")
                    pbar.update(1)
        
        # 计算kappa系数
        kappa = cohen_kappa_score(human_scores, scores, weights='quadratic')
        
        # 记录详细的评分统计
        score_distribution = pd.Series(scores).value_counts().sort_index()
        human_score_distribution = pd.Series(human_scores).value_counts().sort_index()
        
        stats_msg = (f"\n评分统计:\n"
                     f"- 总作文数: {len(scores)}\n"
                     f"- 模型评分分布:\n{score_distribution.to_string()}\n"
                     f"- 人工评分分布:\n{human_score_distribution.to_string()}\n"
                     f"- 模型平均分: {np.mean(scores):.2f}\n"
                     f"- 人工平均分: {np.mean(human_scores):.2f}\n"
                     f"- 加权Kappa系数: {kappa:.4f}")
        
        self.logger.info(stats_msg)
        
        # 创建详细的评分结果
        results = {
            'scores': scores,
            'human_scores': human_scores,
            'evaluations': evaluations,
            'kappa': kappa,
            'model_mean': np.mean(scores),
            'human_mean': np.mean(human_scores)
        }
        
        # 保存评分结果
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """保存评分结果到文件"""
        # 创建输出目录
        os.makedirs(self.out_dir, exist_ok=True)
        
        # 保存详细分数
        scores_df = pd.DataFrame({
            'essay_idx': range(len(results['scores'])),
            'model_score': results['scores'],
            'human_score': results['human_scores'],
            'difference': np.array(results['scores']) - np.array(results['human_scores'])
        })
        scores_df.to_csv(os.path.join(self.out_dir, 'detailed_scores.csv'), index=False)
        
        # 保存评估指标
        metrics_df = pd.DataFrame({
            'metric': ['kappa', 'model_mean', 'human_mean'],
            'value': [
                results['kappa'],
                results['model_mean'],
                results['human_mean']
            ]
        })
        metrics_df.to_csv(os.path.join(self.out_dir, 'metrics.csv'), index=False)
        
        # 保存详细评估信息
        with open(os.path.join(self.out_dir, 'detailed_evaluations.json'), 'w', encoding='utf-8') as f:
            json.dump(results['evaluations'], f, ensure_ascii=False, indent=2)
        
        # 绘制分数分布图
        self._plot_score_distributions(results)
    
    def _plot_score_distributions(self, results: Dict[str, Any]):
        """绘制分数分布比较图"""
        plt.figure(figsize=(12, 6))
        
        # 计算每个分数的频率
        model_counts = pd.Series(results['scores']).value_counts().sort_index()
        human_counts = pd.Series(results['human_scores']).value_counts().sort_index()
        
        # 合并分数范围
        all_scores = sorted(set(list(model_counts.index) + list(human_counts.index)))
        
        # 获取两种分布的数据，对缺失的分数填0
        model_data = [model_counts.get(score, 0) for score in all_scores]
        human_data = [human_counts.get(score, 0) for score in all_scores]
        
        # 绘制双柱状图
        x = range(len(all_scores))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], model_data, width, label='模型评分')
        plt.bar([i + width/2 for i in x], human_data, width, label='人工评分')
        
        plt.xlabel('分数')
        plt.ylabel('频次')
        plt.title('模型评分与人工评分分布对比')
        plt.xticks(x, all_scores)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加Kappa值信息
        plt.figtext(0.5, 0.01, f'加权Kappa系数: {results["kappa"]:.4f}', 
                   ha='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'score_distribution.png'))
        plt.close()

    def save_prompts(self, essays: List[str]):
        """只保存输入prompt作为示例"""
        conversation_dir = os.path.join(self.out_dir, 'conversations')
        os.makedirs(conversation_dir, exist_ok=True)

        for i, essay in enumerate(essays):
            prompt = self.generate_prompt(essay)
            conversation_filename = f"prompt_sample_{i + 1}.txt"
            with open(os.path.join(conversation_dir, conversation_filename), 'w', encoding='utf-8') as f:
                f.write("=== 输入 Prompt ===\n")
                f.write(prompt)

        self.logger.info(f"已保存{len(essays)}篇作文的输入prompt示例到{conversation_dir}")


def load_test_essays(file_path: str, essay_set_id: int) -> Tuple[List[str], List[int]]:
    """加载测试集作文和分数"""
    df = pd.read_excel(file_path)
    
    # 转换为整数类型并筛选指定集合
    df['essay_set'] = df['essay_set'].astype(int)
    df['human_score'] = df['human_score'].astype(int)
    
    # 直接筛选对应essay_set的数据作为测试集
    test_data = df[df['essay_set'] == essay_set_id]
    
    print(f"找到essay_set_{essay_set_id}的测试集数据：{len(test_data)}篇作文")
    
    # 直接返回筛选后的全部数据作为测试集
    test_essays = test_data['essay'].tolist()
    test_scores = test_data['human_score'].tolist()
    
    return test_essays, test_scores

def get_prompt(dataset: str, essay_set_value: int) -> str:
    """获取作文提示"""
    file_path = os.path.join('..', 'Data', dataset, 'prompts', f'Essaay Set #{essay_set_value}.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt

def get_score_range(dataset: str, essay_set_value: int) -> Tuple[int, int]:
    """获取评分范围"""
    file_path = os.path.join('..', 'Data', dataset, 'score range.xlsx')
    df = pd.read_excel(file_path)
    
    # 确保essay_set列为整数类型
    df['essay_set'] = df['essay_set'].astype(int)
    
    # 获取score_range并解析为元组
    score_range_str = df[df['essay_set'] == essay_set_value]['score_range'].values[0]
    
    # 解析不同格式的score_range
    if isinstance(score_range_str, str) and 'points' in score_range_str:
        # 移除 "points" 并分割数字
        numbers = score_range_str.replace('points', '').strip().split('-')
        return (int(numbers[0]), int(numbers[1]))
    elif isinstance(score_range_str, str):
        score_range_str = score_range_str.strip('()[]{}').split(',')
        return (int(score_range_str[0]), int(score_range_str[1]))
    elif isinstance(score_range_str, (tuple, list)):
        return (int(score_range_str[0]), int(score_range_str[1]))
    else:
        raise ValueError(f"Unexpected score_range format: {score_range_str}")

def main():
    """主函数"""
    # 命令行参数
    parser = argparse.ArgumentParser(description="使用最佳评分标准对测试集进行评分")
    parser.add_argument('--dataset', default='ASAP', help="数据集名称")
    parser.add_argument('--essay_set', default=1, type=int, help="作文集编号")
    parser.add_argument('--model', default='glm', help="使用的模型")
    parser.add_argument('--temperature', default=0.3, type=float, help="模型温度参数")
    parser.add_argument('--output_dir', default='cross-model-glm(doubao-rubric)', help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f'essay_set_{args.essay_set}', timestamp)
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存配置信息
    config = vars(args)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 获取作文提示和评分范围
    essay_prompt = get_prompt(args.dataset, args.essay_set)
    score_range = get_score_range(args.dataset, args.essay_set)
    
    # 加载测试集 - 使用相对路径
    file_path = os.path.join('..', 'Data', 'Score(test set).xlsx')
    test_essays, test_scores = load_test_essays(file_path, args.essay_set)
    
    print(f"加载了{len(test_essays)}篇测试集作文")
    print(f"评分范围: {score_range}")
    
    # 加载最佳评分标准文件 - 使用相对路径
    best_template_path = os.path.join('..', 'Data', 'Rubric.xlsx')
    
    # 使用最佳评分标准进行评分
    scorer = BestTemplateScorer(
        model=args.model,
        temperature=args.temperature,
        scoring_range=score_range,
        essay_prompt=essay_prompt,
        best_template_path=best_template_path,
        essay_set_id=args.essay_set,
        out_dir=out_dir
    )

    # 保存所有评分prompt
    scorer.save_prompts(test_essays)

    print("开始评分...")
    results = scorer.evaluate(test_essays, test_scores)
    
    # 输出结果
    print("\n评分结果:")
    print(f"测试集大小: {len(test_essays)}篇作文")
    print(f"加权Kappa系数: {results['kappa']:.4f}")
    print(f"模型平均分: {results['model_mean']:.2f}")
    print(f"人工平均分: {results['human_mean']:.2f}")
    print(f"详细结果已保存到: {out_dir}")

    # 输出当前使用的最佳评分标准（只展示前200个字符，避免输出过长）
    scoring_criteria = scorer.template["scoring_criteria"]
    preview = scoring_criteria[:200] + "..." if len(scoring_criteria) > 200 else scoring_criteria
    print("\n当前使用的最佳评分标准预览:")
    print(f"essay_set_{args.essay_set}的评分标准: {preview}")

    # 保存完整的评分标准到文件
    with open(os.path.join(out_dir, 'used_scoring_criteria.txt'), 'w', encoding='utf-8') as f:
        f.write(scoring_criteria)
    print(f"完整的评分标准已保存到: {os.path.join(out_dir, 'used_scoring_criteria.txt')}")

if __name__ == "__main__":
    main()