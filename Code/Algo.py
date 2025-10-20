import concurrent.futures
import os
import json
from tqdm import tqdm
import utils
from sklearn.metrics import cohen_kappa_score
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import random
import pandas as pd
from utils import parse_json_response
from utils import ThreadSafeLogger
from utils import _CACHE
import matplotlib.pyplot as plt
import time
from typing import Optional


class PromptTemplate:
    """
        评分prompt模板类
        负责管理和生成用于评分的prompt，支持模板的变异和交叉操作
        """

    def __init__(self, base_instruction: str, scoring_criteria: str, output_format: str, args=None):
        self.base_instruction = base_instruction
        self.scoring_criteria = scoring_criteria
        self.output_format = output_format
        self.args = args  # 添加 args 属性

    def mutate(self) -> 'PromptTemplate':
        """
        对当前评分标准进行小幅变异
        通过LLM生成评分标准的微小变体，保持核心含义不变

        Returns:
            新的PromptTemplate实例，包含变异后的评分标准
        """
        mutation_prompt = f"""
        You are an expert in essay assessment. Below is a scoring rubric with different score points:

        {self.scoring_criteria}

        Create an incrementally enhanced version of this rubric. Your goal is to make SMALL, STABLE improvements while maintaining consistency with the original structure.

        Requirements:

        1. Stability Guidelines:
            - Keep 50-70% of the original content unchanged, focusing changes on the most ambiguous or problematic criteria
            - Make only incremental changes to existing criteria
            - Any new additions must directly support and clarify existing criteria
            - Preserve the core evaluation framework and scoring philosophy
            - Replace vague terms like "good" or "adequate" with specific descriptors

        2. Consistency Focus:
            - Ensure new criteria align with existing evaluation patterns
            - Maintain consistent terminology and language style across all levels
            - Preserve the relative importance and weighting of each criterion
            - Ensure smooth logical progression that maintains the original score level ordering and hierarchy
            - Preserve clear distinctions between adjacent score levels in their established sequence

        3. Quality Control:
            - New criteria must be objective and observable
            - Avoid dramatic shifts in evaluation emphasis
            - Ensure criteria are applicable across different essay topics and styles

        4. Avoid These Changes:
            - Do not alter the fundamental scoring philosophy or approach
            - Do not change the score range, level structure, or number of levels
            - Do not introduce criteria that require specialized subject knowledge
            - Do not create overlapping, contradictory, or redundant requirements
            - Do not add criteria that are impossible to measure objectively
            
        Remember: Focus on SMALL, STABLE improvements rather than major overhauls.

        Output the improved rubric as a valid JSON response, following exactly the same format as the input rubric."""

        # Output the improved rubric following the same format and structure as the input rubric, but with substantially enhanced content.
        #清空缓存
        _CACHE.clear_cache()
        new_criteria = utils.chat(self.args['model'], mutation_prompt)[0]
        return PromptTemplate(self.base_instruction, new_criteria, self.output_format, self.args)

    # 这个helper函数的作用是将评分标准字符串转换成列表，方便后续的交叉操作

    def crossover(self, other: 'PromptTemplate') -> 'PromptTemplate':
        """
        将两个评分标准进行交叉组合
        通过LLM将当前模板和另一个模板的评分标准融合

        Args:
            other: 另一个PromptTemplate实例
        Returns:
            新的PromptTemplate实例，包含交叉后的评分标准
        """
        crossover_prompt = f"""
        You are an expert in essay assessment systems. You need to carefully combine two scoring rubrics to create a more stable and effective one.

        Given two scoring rubrics:
        Rubric A: {self.scoring_criteria}
        Rubric B: {other.scoring_criteria}

        Create a new rubric following these stability-focused guidelines:

        1. Conservative Combination Rules:
            - Keep 50-70% of the criteria from the better-performing rubric
            - Only adopt clearly superior elements from the other rubric
            - Maintain exact scoring ranges and level structure from the original rubrics
            - Preserve the fundamental evaluation approach of the primary rubric

        2. Integration Priorities:
            - Prefer specific, observable criteria over general descriptions
            - Choose criteria that are clear and consistently applicable
            - Maintain clear distinctions between different score levels
            - Focus on writing characteristics that apply across different topics and styles

        3. Stability Requirements:
            - Ensure logical progression that maintains the original score level ordering and hierarchy
            - Use consistent terminology and language style throughout
            - Keep criteria weights balanced and stable
            - Preserve the relative importance of each evaluation dimension

        4. Quality Control:
        - New criteria must be objective and observable
        - Verify compatibility and consistency of combined elements
        - Avoid creating overlapping, contradictory, or redundant requirements
        - Ensure the combined rubric remains applicable across different essay topics

        5. Avoid These Changes:
        - Do not alter the fundamental scoring philosophy or approach
        - Do not change the score range, level structure, or number of levels
        - Do not introduce criteria that require specialized subject knowledge
        - Do not create dramatic shifts in evaluation emphasis

        Remember: 1. Prioritize STABILITY and CONSISTENCY over innovation.
                2. Please output the complete rubric following the same format as the input rubrics.
                3. Focus on creating a practical, reliable scoring tool.
                
        Output the combined rubric as a valid JSON response, following exactly the same format as the input rubrics."""
        # Output the combined rubric following the same format as the input rubrics. Focus on creating a clear, practical, and effective scoring rubric.

        #清空缓存
        _CACHE.clear_cache()

        new_criteria = utils.chat(self.args['model'], crossover_prompt)[0]

        return PromptTemplate(self.base_instruction, new_criteria, self.output_format, self.args)

    # generate_prompt方法是输入一篇作文和评分标准，输出完整的评分prompt。
    def generate_prompt(self, essay: str, essay_prompt: str) -> str:
        """生成完整的评分prompt"""
        return f"""
        {self.base_instruction}

        Writing Task:
        {essay_prompt}

        Essay to evaluate:
        {essay}

        Scoring Criteria:
        {self.scoring_criteria}

        {self.output_format}
        """


class ScoringSystem:
    """评分系统基类"""

    def __init__(self, args: Dict[str, Any], scoring_range: Tuple[int, int],
                 essay_prompt: str, out_dir: str):
        self.args = args
        self.score_range = scoring_range
        self.essay_prompt = essay_prompt
        self.out_dir = out_dir
        self.logger = ThreadSafeLogger(essay_set_id=args.get('essay_set'))  # 添加 logger 实例

class BaseScorer(ScoringSystem):
    """基础评分系统"""

    def __init__(self, args: Dict[str, Any], scoring_range: Tuple[int, int],
                 essay_prompt: str, out_dir: str):
        super().__init__(args, scoring_range, essay_prompt, out_dir)

        output_format = f'''Return JSON in this format:
                {{"score": <integer between {scoring_range[0]} and {scoring_range[1]}>}}
                Note: Return only the JSON object without any additional text.'''

        self.template = PromptTemplate(
            base_instruction="You are a secondary school English teacher. Please evaluate the following essay:",
            scoring_criteria=args['initial_rubric'],
            output_format=output_format
        )

    # 在 BaseScorer 类的 score_essay 方法中需要确保 score_range 的类型处理
    def score_essay(self, essay: str) -> int:
        try:
            prompt = self.template.generate_prompt(essay, self.essay_prompt)
            response = utils.chat(self.args['model'], prompt, temperature=self.args['temperature'])[0]
            # 使用 score_range[0] 作为最小分值
            result = parse_json_response(response, min_score=self.score_range[0])
            min_score, max_score = int(self.score_range[0]), int(self.score_range[1])
            score = max(min(result['score'], max_score), min_score)

            return score
        except Exception as e:
            self.logger.error(f"Error parsing response: {response}\nError details: {str(e)}")
            # 返回当前评分标准的最小值，而不是0
            return self.score_range[0]

    def evaluate(self, essays: List[str], human_scores: List[int]) -> Dict[str, Any]:
        self.logger.section("Base Scoring System Evaluation")
        self.logger.info(
            f"Configuration:\n"
            f"- Model: {self.args['model']}\n"
            f"- Temperature: {self.args['temperature']}\n"
            f"- Score Range: {self.score_range}"
        )

        scores = []
        total_essays = len(essays)
        for idx, essay in enumerate(tqdm(essays, desc="Base scoring")):
            score = self.score_essay(essay)
            scores.append(score)
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Processed {idx + 1}/{total_essays} essays")

        kappa = cohen_kappa_score(human_scores, scores, weights='quadratic')

        # 记录详细的评分统计
        score_distribution = pd.Series(scores).value_counts().sort_index()
        stats_msg = (f"\nScoring Statistics:\n"
                     f"- Total Essays: {len(scores)}\n"
                     f"- Score Distribution:\n{score_distribution.to_string()}\n"
                     f"- Mean Score: {np.mean(scores):.2f}\n"
                     f"- Median Score: {np.median(scores):.2f}\n"
                     f"- Kappa Score: {kappa:.4f}")

        self.logger.info(stats_msg)

        return {
            'scores': scores,
            'kappa': kappa
        }


class EvolvedScorer(ScoringSystem):
    """使用进化算法优化的评分系统"""

    def __init__(self, args: Dict[str, Any], scoring_range: Tuple[int, int],
                 essay_prompt: str, out_dir: str):
        super().__init__(args, scoring_range, essay_prompt, out_dir)

        self.population_size = 10
        self.generations = args.get('generations', 20)
        self.mutation_prob = 0.3
        self.crossover_prob = 0.7
        self.max_workers = min(8, (os.cpu_count() or 1) * 4)
        self.best_kappas_per_gen = []
        
        # 最后初始化种群
        self.population = self._initialize_population(args['initial_rubric'])

    def _initialize_population(self, initial_rubric: str) -> List[PromptTemplate]:
        """初始化种群"""
        # 添加日志记录
        self.logger.section("Population Initialization")
        self.logger.info(f"Starting population initialization with size: {self.population_size}")
        
        population = []

        output_format = f'''
        IMPORTANT JSON FORMATTING RULES:
        1. Ensure all properties within objects are separated by commas
        2. Do not add a comma after the last property in an object
        3. All string values must be properly quoted

        Output your complete evaluation of all essays provided above as a valid JSON response using this exact structure:
                {{
                    "evaluations": [
                        {{
                            "essay_number": <essay number>,
                            "criteria_analysis": [
                                {{
                                    "criterion": "criterion name from rubric",
                                    "criterion_level": <integer between {self.score_range[0]} and {self.score_range[1]}>
                                }},            
                                // Add more criteria as needed
                            ],
                            "score": <integer between {self.score_range[0]} and {self.score_range[1]}>,
                            "justification": "Explain how criteria led to final score"
                        }}
                    ]
                }}'''

        base_instruction = '''You are an experienced secondary school English teacher. Please evaluate the following essays following this systematic process:

                Step 1: Criterion-Based Assessment
                For each essay and each criterion in the scoring rubric, you must:
                - Determine the score level at which this criterion is met
                - Identify any language or structural elements that support your evaluation

                Step 2: Consider the overall response
                For each essay:
                - Review all criteria evaluations together
                - Determine if the evidence consistently supports a particular score level
                - Identify and resolve any conflicts between criteria scores
                - Consider how different criteria interact and influence each other

                Step 3: Provide final score and justification that references the specific criteria

                '''

        # 记录创建基础模板
        self.logger.info("Creating base template with initial rubric")
        base_template = PromptTemplate(
            base_instruction=base_instruction,
            scoring_criteria=initial_rubric,
            output_format=output_format,
            args=self.args  # 传递 args 参数
        )

        # 将基础模板添加到种群中并记录详细内容
        population.append(base_template)
        self.logger.info(f"Template 1/{self.population_size}: Base template added to population")
        self.logger.info(f"Template 1 Scoring Criteria:\n{base_template.scoring_criteria}")
        
        # 使用tqdm创建进度条
        print("\nGenerating initial population through mutation...")
        # 通过对基础模板进行变异(mutate)生成剩余的种群成员
        for i in tqdm(range(self.population_size - 1), desc="Population initialization"):
            # 记录变异进度
            template_num = i + 2  # 模板编号
            self.logger.info(f"Generating template {template_num}/{self.population_size} through mutation...")
            mutated_template = base_template.mutate()
            population.append(mutated_template)
            
            # 记录变异后模板的详细内容
            self.logger.info(f"Template {template_num}/{self.population_size}: Mutation completed")
            self.logger.info(f"Template {template_num} Scoring Criteria:\n{mutated_template.scoring_criteria}")

        self.logger.info(f"Population initialization completed. Total templates: {len(population)}")
        return population

    # 对多个模板进行并行评估
    def parallel_evaluate_templates(self, templates: List[PromptTemplate],
                                    validation_essays: List[str],
                                    human_scores: List[int]) -> List[Tuple[PromptTemplate, Dict]]:
        """并行评估多个模板"""
        # 预分配结果列表
        results = [None] * len(templates)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 记录每个future对应的原始索引
            future_to_idx = {
                executor.submit(self.evaluate_template, template, validation_essays, human_scores): idx
                for idx, template in enumerate(templates)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    stats = future.result()
                    results[idx] = (templates[idx], stats)  # 保持原始顺序
                except Exception as e:
                    self.logger.error(f"Template evaluation failed for template {idx}: {str(e)}")
                    # 在失败时也保持位置
                    results[idx] = (templates[idx], {'mean_kappa': 0.0, 'explanations': []})

        # 过滤掉None值，确保返回有效结果
        return [r for r in results if r is not None]

    def parallel_mutation(self, parents: List[PromptTemplate]) -> List[PromptTemplate]:
        """并行执行变异操作"""
        self.logger.info(f"Starting parallel mutation for {len(parents)} templates")
        results = [None] * len(parents)  # 预分配结果列表

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._safe_mutate, parent): idx
                for idx, parent in enumerate(parents)
            }

            with tqdm(total=len(parents), desc="Mutation") as pbar:
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                        self.logger.info(f"Completed mutation for template {idx}/{len(parents)}")
                    except Exception as e:
                        self.logger.error(f"Mutation failed for template {idx}: {str(e)}")
                        results[idx] = parents[idx]  # 失败时保持原模板
                        self.logger.info(f"Using original template for {idx} due to mutation failure")
                    pbar.update(1)

        # 过滤掉None值，确保返回有效结果
        valid_results = [r for r in results if r is not None]
        self.logger.info(f"Parallel mutation completed. Generated {len(valid_results)} valid templates")
        return valid_results

    def _safe_mutate(self, template: PromptTemplate, max_retries: int = 3) -> PromptTemplate:
        """安全的变异操作，包含重试机制和详细日志记录"""
        self.logger.info(f"Starting mutation operation")
        self.logger.info(f"Original scoring criteria before mutation:\n{template.scoring_criteria}")
        
        for attempt in range(max_retries):
            try:
                # 执行变异操作
                mutated_template = template.mutate()
                
                # 记录变异后的评分标准
                self.logger.info(f"Mutation completed successfully (attempt {attempt+1}/{max_retries})")
                self.logger.info(f"New scoring criteria after mutation:\n{mutated_template.scoring_criteria}")
                
                return mutated_template
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Mutation failed after {max_retries} attempts: {str(e)}")
                    raise e
                self.logger.warning(f"Mutation attempt {attempt+1} failed: {str(e)}. Retrying...")
                time.sleep(3)  # 重试前短暂等待

    def parallel_crossover(self, parents: List[PromptTemplate]) -> List[PromptTemplate]:
        """并行执行交叉操作"""
        if len(parents) < 2:
            self.logger.info(f"Not enough parents for crossover (only {len(parents)}). Returning copies.")
            return parents.copy()

        pairs = [(parents[i], parents[i + 1]) for i in range(0, len(parents) - 1, 2)]
        self.logger.info(f"Starting parallel crossover for {len(pairs)} pairs of templates")
        results = [None] * len(pairs)  # 预分配结果列表

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._safe_crossover, pair[0], pair[1]): idx
                for idx, pair in enumerate(pairs)
            }

            with tqdm(total=len(pairs), desc="Crossover") as pbar:
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                        self.logger.info(f"Completed crossover for pair {idx}/{len(pairs)}")
                    except Exception as e:
                        self.logger.error(f"Crossover failed for pair {idx}: {str(e)}")
                        results[idx] = pairs[idx][0]  # 失败时保持第一个父代模板
                        self.logger.info(f"Using first parent template for pair {idx} due to crossover failure")
                    pbar.update(1)

        # 过滤掉None值，确保返回有效结果
        valid_results = [r for r in results if r is not None]
        self.logger.info(f"Parallel crossover completed. Generated {len(valid_results)} valid templates")
        return valid_results

    def _safe_crossover(self, template1: PromptTemplate, template2: PromptTemplate, 
                        max_retries: int = 3) -> PromptTemplate:
        """安全的交叉操作，包含重试机制和详细日志记录"""
        self.logger.info(f"Starting crossover operation between two templates")
        self.logger.info(f"Parent template 1 scoring criteria:\n{template1.scoring_criteria}")
        self.logger.info(f"Parent template 2 scoring criteria:\n{template2.scoring_criteria}")
        
        for attempt in range(max_retries):
            try:
                # 执行交叉操作
                crossed_template = template1.crossover(template2)
                
                # 记录交叉后的评分标准
                self.logger.info(f"Crossover completed successfully (attempt {attempt+1}/{max_retries})")
                self.logger.info(f"New scoring criteria after crossover:\n{crossed_template.scoring_criteria}")
                
                return crossed_template
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Crossover failed after {max_retries} attempts: {str(e)}")
                    raise e
                self.logger.warning(f"Crossover attempt {attempt+1} failed: {str(e)}. Retrying...")
                time.sleep(3)  # 重试前短暂等待

    # 用于evaluate_template的重试机制
    def _safe_chat_with_explanation(self, prompt: str) -> tuple[str, bool]:
        """
        带解释的安全API调用
        Returns:
            tuple[str, bool]: (response, is_success)
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = utils.chat(
                    self.args['model'],
                    prompt,
                    temperature=self.args['temperature']
                )[0]
                return response, True
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"(在_safe_chat_with_explanation方法中出错)Failed after {max_retries} attempts: {str(e)}")
                    return "", False
                time.sleep(1)  # 简单延迟重试

    # 对单个模板在验证集中进行评估，是evaluate_template的子函数
    def evaluate_template(self, template: PromptTemplate,
                        validation_essays: List[str],
                        human_scores: List[int],
                        num_trials: int = 1) -> Dict[str, float]:
        """评估单个模板"""
        scores = [None] * len(validation_essays)  # 预分配列表
        explanations = [None] * len(validation_essays)

        with concurrent.futures.ThreadPoolExecutor(max_workers = min(8, len(validation_essays))) as executor:
            future_to_idx = {
                executor.submit(self._score_single_essay, template, essay): idx
                for idx, essay in enumerate(validation_essays)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    score, explanation = future.result()
                    scores[idx] = score
                    explanations[idx] = explanation
                except Exception as e:
                    self.logger.error(f"Failed to score essay {idx}")
                    scores[idx] = self.score_range[0]
                    explanations[idx] = {}

        # 确保所有位置都有值
        scores = [s if s is not None else self.score_range[0] for s in scores]

        # 计算kappa系数
        kappa = cohen_kappa_score(human_scores, scores, weights='quadratic')

        return {
            'mean_kappa': kappa,
            'explanations': explanations
        }

    def _score_single_essay(self, template: PromptTemplate, essay: str) -> tuple[int, dict]:
        """评分单篇作文的辅助函数"""
        try:
            prompt = template.generate_prompt(essay, self.essay_prompt)
            response, success = self._safe_chat_with_explanation(prompt)

            if not success:
                return self.score_range[0], {}

            parsed_result = utils.parse_json_response(response, min_score=self.score_range[0])

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

            return score, explanation

        except Exception as e:
            # 添加更详细的错误日志，包含原始响应和解析后的结果
            self.logger.error(
                f"Error scoring essay in 评估单篇作文:\n"
                f"Error: {str(e)}\n"
                f"Raw Response: {response}\n"
                f"Parsed Result: {parsed_result if 'parsed_result' in locals() else 'Not parsed'}"
            )
            return self.score_range[0], {}

    def should_accept(self, old_stats: Dict[str, float],
                      new_stats: Dict[str, float]) -> bool:
        """决定是否接受新模板"""
        return new_stats['mean_kappa'] > old_stats['mean_kappa']
        # 只要新模板的kappa值大于旧模板的kappa值，就接受新模板

    def plot_evolution_progress(self):
        """绘制进化过程中每代的最佳kappa值"""
        plt.figure(figsize=(10, 6))
        generations = range(1, len(self.best_kappas_per_gen) + 1)

        plt.plot(generations, self.best_kappas_per_gen, 'b-', marker='o')
        plt.title('Evolution Progress: Best Kappa per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Best Kappa Score')
        plt.grid(True)

        # 添加数值标签
        for i, kappa in enumerate(self.best_kappas_per_gen):
            plt.annotate(f'{kappa:.4f}',
                         (i + 1, kappa),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')

        # 保存图表
        plt.savefig(os.path.join(self.out_dir, 'evolution_progress.png'))
        plt.close()

    def           evolve(self, train_essays: List[str], train_scores: List[int],
               valid_essays: List[str], valid_scores: List[int]) -> PromptTemplate:
        """使用训练集优化评分标准，使用验证集评估防止过拟合"""
        start_time = datetime.now()

        # 打印参数
        print("\n" + "=" * 50)
        print("Model Configuration:")
        print(f"- Model: {self.args['model']}")
        print(f"- Temperature: {self.args['temperature']}")
        print(f"- Population Size: {self.population_size}")
        print(f"- Generations: {self.generations}")
        print(f"- Thread Pool Size: {self.max_workers}")
        print("=" * 50 + "\n")

        # 记录日志
        self.logger.section("Evolution Process Started")
        self.logger.info(
            f"Evolution Configuration:\n"
            f"- Population Size: {self.population_size}\n"
            f"- Generations: {self.generations}\n"
            f"- Mutation Probability: {self.mutation_prob}\n"
            f"- Crossover Probability: {self.crossover_prob}\n"
            f"- Thread Pool Size: {self.max_workers}"
        )

        best_template = self.population[0]
        # 使用验证集评估初始模板
        best_stats = self.evaluate_template(best_template, valid_essays, valid_scores)
        self.logger.info(f"Initial template performance: Kappa = {best_stats['mean_kappa']:.4f}")

        for generation in range(self.generations):
            # 清空上一代的缓存
            _CACHE.clear_cache()
            self.logger.info(f"Cache cleared after generation {generation + 1}")

            # 打印当前代数
            print(f"\nGeneration {generation + 1}/{self.generations}")
            print("-" * 30)

            # 记录当前代数日志
            self.logger.section(f"Generation {generation + 1}/{self.generations}")

            # 使用训练集并行评估所有模板
            template_stats = self.parallel_evaluate_templates(
                self.population, train_essays, train_scores
            )
            template_stats.sort(key=lambda x: x[1]['mean_kappa'], reverse=True)
            current_best = template_stats[0]

            # 使用验证集评估当前最佳模板
            current_valid_stats = self.evaluate_template(
                current_best[0], valid_essays, valid_scores
            )

            # 记录当代最佳表现
            self.logger.info(
                f"\nGeneration Best Performance:\n"
                f"- Training Kappa: {current_best[1]['mean_kappa']:.4f}\n"
                f"- Validation Kappa: {current_valid_stats['mean_kappa']:.4f}\n"
                f"- 当代最佳评分标准(Template Criteria):\n{current_best[0].scoring_criteria}"
            )

            # 基于验证集性能决定是否接受新模板
            if self.should_accept(best_stats, current_valid_stats):
                improvement = current_valid_stats['mean_kappa'] - best_stats['mean_kappa']
                best_template = current_best[0]
                best_stats = current_valid_stats
                self.logger.info(
                    f"New best template found!\n"
                    f"- Improvement: +{improvement:.4f}\n"
                    f"- New Validation Kappa: {best_stats['mean_kappa']:.4f}\n"
                    f"- New Best Template Criteria:\n{best_template.scoring_criteria}"
                )

            # 记录当代最佳kappa值
            self.best_kappas_per_gen.append(current_valid_stats['mean_kappa'])

            # 选择父代和生成新一代的逻辑
            parents = [t for t, _ in template_stats[:self.population_size // 2]]
            new_population = parents.copy()
            
            # 记录保留的父代模板
            self.logger.section(f"Generation {generation + 1} Population Update")
            self.logger.info(f"Keeping top {len(parents)} templates as parents")
            for i, parent in enumerate(parents):
                self.logger.info(f"Parent {i+1}/{len(parents)}: Kappa = {template_stats[i][1]['mean_kappa']:.4f}")
            
            # 记录交叉操作
            if random.random() < self.crossover_prob:
                self.logger.info(f"Performing crossover operations")
                crossed = self.parallel_crossover(parents)
                
                # 记录每个交叉生成的模板
                self.logger.info(f"Generated {len(crossed)} templates through crossover")
                for i, template in enumerate(crossed):
                    self.logger.info(f"Crossover template {i+1}/{len(crossed)}:")
                    self.logger.info(f"Scoring criteria:\n{template.scoring_criteria}")
                
                new_population.extend(crossed)
                self.logger.info(f"Population size after crossover: {len(new_population)}/{self.population_size}")
            
            # 记录变异操作
            if len(new_population) < self.population_size:
                remaining_slots = self.population_size - len(new_population)
                self.logger.info(f"Need {remaining_slots} more templates through mutation")
                
                while len(new_population) < self.population_size:
                    mutation_candidates = random.sample(
                        parents,
                        k=min(self.max_workers, len(parents))
                    )
                    
                    self.logger.info(f"Selected {len(mutation_candidates)} parent templates for mutation")
                    mutated = self.parallel_mutation(mutation_candidates)
                    
                    # 记录将要添加的变异模板
                    to_add = mutated[:self.population_size - len(new_population)]
                    self.logger.info(f"Adding {len(to_add)} mutation templates to population")
                    
                    for i, template in enumerate(to_add):
                        self.logger.info(f"Mutation template {i+1}/{len(to_add)}:")
                        self.logger.info(f"Scoring criteria:\n{template.scoring_criteria}")
                    
                    new_population.extend(to_add)
                
                self.logger.info(f"Population size after mutation: {len(new_population)}/{self.population_size}")
            
            self.population = new_population[:self.population_size]
            self.logger.info(f"Final population size for next generation: {len(self.population)}")

        # self.logger.section("Evolution Complete")
        self.logger.info(f"Final Results:\n" +
                         f"- Best Kappa: {best_stats['mean_kappa']:.4f}\n" +
                         f"- Best Template Criteria:\n{best_template.scoring_criteria}")

        # 确保在结束时保存最佳模板
        self.best_template = best_template
        self.logger.info(f"Evolution process completed.")
        self.logger.info(f"Best template saved with kappa: {best_stats['mean_kappa']:.4f}")
        
        # 绘制进化进度图
        self.plot_evolution_progress()
        
        return best_template

    def evaluate(self, test_essays: List[str], test_scores: List[int]) -> Dict[str, Any]:
        """在测试集上评估最终性能"""
        self.logger.info("Starting test set evaluation...")

        # 预分配列表，确保顺序
        scores = [self.score_range[0]] * len(test_essays)
        evaluations = [{}] * len(test_essays)  # 存储完整的评分结果

        # 使用线程池并行评分
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 为每篇作文创建评分任务，使用_score_single_essay方法获取详细评分
            future_to_idx = {
                executor.submit(self._score_single_essay, self.best_template, essay): idx
                for idx, essay in enumerate(test_essays)
            }

            # 使用tqdm创建进度条
            with tqdm(total=len(test_essays), desc="Test set scoring") as pbar:
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        score, evaluation = future.result()  # 获取分数和评估结果
                        scores[idx] = score
                        evaluations[idx] = evaluation  # 保存完整的评估结果
                    except Exception as e:
                        self.logger.error(f"Failed to score essay {idx}")
                        scores[idx] = self.score_range[0]
                        evaluations[idx] = {}
                    pbar.update(1)

        # 计算kappa系数
        kappa = cohen_kappa_score(test_scores, scores, weights='quadratic')

        # 记录详细的评分统计
        score_distribution = pd.Series(scores).value_counts().sort_index()
        stats_msg = (f"\nScoring Statistics:\n"
                     f"- Total Essays: {len(scores)}\n"
                     f"- Score Distribution:\n{score_distribution.to_string()}\n"
                     f"- Mean Score: {np.mean(scores):.2f}\n"
                     f"- Median Score: {np.median(scores):.2f}\n"
                     f"- Kappa Score: {kappa:.4f}")

        self.logger.info(f"Test set evaluation completed. {stats_msg}")

        # 保存evaluations结果
        output_file = os.path.join(self.out_dir, 'evaluations.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Evaluations saved to: {output_file}")

        return {
            'scores': scores,
            'evaluations': evaluations,
            'kappa': kappa
        }

    # 针对evaluate的重试机制
    def _safe_chat_score(self, prompt: str) -> tuple[int, bool]:
        """
        仅返回分数的安全API调用
        Returns:
            tuple[int, bool]: (score, is_success)
        """
        max_retries = 2  # 测试集评分可以用更少的重试次数
        for attempt in range(max_retries):
            try:
                response = utils.chat(
                    self.args['model'],
                    prompt,
                    temperature=self.args['temperature']
                )[0]
                return response, True
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                    return "", False
                time.sleep(0.5)  # 更短的重试延迟

    # 添加新的辅助方法
    def score_single_essay(self, essay: str, template: PromptTemplate) -> int:
        """对单篇作文进行评分"""
        try:
            prompt = template.generate_prompt(essay, self.essay_prompt)
            response = utils.chat(self.args['model'], prompt,
                                  temperature=self.args['temperature'])[0]

            result = parse_json_response(response)

            if 'evaluations' in result:
                score = result['evaluations'][0]['score']
            elif 'score' in result:
                score = result['score']
            else:
                self.logger.warning(f"Invalid response format: {response}")
                score = self.score_range[0]

            return score

        except Exception as e:
            # 添加更详细的错误日志，包含原始响应和解析结果
            self.logger.error(
                f"Error scoring essay in evaluate 评估单篇作文（test set）:\n"
                f"Error: {str(e)}\n"
                f"Raw Response: {response}\n"
                f"Parsed Result: {result if 'result' in locals() else 'Not parsed'}"
            )
            return self.score_range[0]


