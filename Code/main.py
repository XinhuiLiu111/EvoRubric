# main8.py
# 主程序文件，用于协调评分系统的整体流程，包括数据加载、评分执行和结果保存

import os
from datetime import datetime
import json
import argparse
from getData7 import GetData
from Algo import BaseScorer, EvolvedScorer
import pandas as pd


def get_args():
    """
    解析命令行参数,优先级最高
    Returns:
        解析后的参数对象，包含：
        - dataset: 数据集名称，默认'ASAP'
        - essay_set: 作文集编号，默认1
        - model: 使用的模型，默认'glm'
        - temperature: 模型温度参数，默认0.3
        - generations: 进化算法迭代次数，默认10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ASAP')
    parser.add_argument('--essay_set', default=1, type=int)
    parser.add_argument('--model', default='glm',
                        choices=['glm', 'deepseek', 'qwen', 'doubao'],
                        help='选择使用的模型，可用选项: glm, deepseek, qwen, doubao')
    parser.add_argument('--temperature', default=0.3, type=float)
    parser.add_argument('--generations', default=5, type=int,
                        help='Number of generations for evolution')

    return parser.parse_args()


def main():
    """
    主函数，执行完整的评分流程：
    1. 初始化配置和输出目录
    2. 加载数据集和评分标准
    3. 执行基础评分和进化评分
    4. 保存评分结果和详细记录
    """
    # 初始化
    args = get_args()
    start_time = datetime.now()

    # 创建输出目录
    out_dir = os.path.join('out', f'essay_set_{args.essay_set}',
                           f'{start_time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(out_dir, exist_ok=True)

    # 记录配置
    config = vars(args)
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 获取数据
    data = GetData(args.dataset, args.essay_set)
    # 获取项目根目录路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, 'Data', args.dataset, 'training_set_rel3.xlsx')
    train_set, valid_set, test_set = data.get_dataset(file_path)

    # # 保存测试集的作文内容和分数
    # test_essays_scores = test_set[['essay', 'domain1_score']]
    # save_path = os.path.join(out_dir, 'test_essays_with_scores.xlsx')
    # test_essays_scores.to_excel(save_path, index=False)
    # print(f"\n测试集的所有作文及分数已保存到 {save_path}")

    # 获取作文提示、分数范围和评分标准
    essay_prompt = data.get_prompt()
    score_range = data.get_score_range()
    initial_rubric = data.get_rubric()

    # 添加评分标准到配置
    config['initial_rubric'] = initial_rubric

    # 执行基础评分
    print("Running base scoring...")
    base_scorer = BaseScorer(config, score_range, essay_prompt, out_dir)
    base_results = base_scorer.evaluate(test_set['essay'], test_set['domain1_score'])
    base_kappa = base_results['kappa']
    print(f"Base Kappa: {base_kappa:.4f}")

    # 执行进化评分
    print("Running evolved scoring...")
    evolved_scorer = EvolvedScorer(config, score_range, essay_prompt, out_dir)

    best_template = evolved_scorer.evolve(
        train_essays=train_set['essay'],
        train_scores=train_set['domain1_score'],
        valid_essays=valid_set['essay'],
        valid_scores=valid_set['domain1_score']
    )

    # 测试集评估
    print("\nEvaluating on test set...")
    evolved_results = evolved_scorer.evaluate(test_set['essay'], test_set['domain1_score'])
    evolved_kappa = evolved_results['kappa']

    # 3. 结果比较
    print(f"\nResults Comparison:")
    print(f"Base Kappa: {base_kappa:.4f}")
    print(f"Evolved Kappa: {evolved_kappa:.4f}")
    print(f"Improvement: {evolved_kappa - base_kappa:.4f}")

    # 确保所有数组长度一致
    min_length = min(
        len(test_set['essay']),
        len(test_set['domain1_score']),
        len(base_results['scores']),
        len(evolved_results['scores'])
    )

    scoring_df = pd.DataFrame({
        'essay': test_set['essay'][:min_length],
        'human_score': test_set['domain1_score'][:min_length],
        'base_score': base_results['scores'][:min_length],
        'evolved_score': evolved_results['scores'][:min_length]
    })

    # 添加评估指标作为单独的行或保存在另一个文件中
    metrics_df = pd.DataFrame({
        'metric': ['base_kappa', 'evolved_kappa', 'improvement'],
        'value': [
            base_results['kappa'],
            evolved_results['kappa'],
            evolved_results['kappa'] - base_results['kappa']
        ]
    })

    # 保存详细分数
    scoring_df.to_excel(os.path.join(out_dir, 'detailed_scores.xlsx'), index=False)
    # 保存评估指标
    metrics_df.to_excel(os.path.join(out_dir, 'evaluation_metrics.xlsx'), index=False)

    # 记录总运行时间
    end_time = datetime.now()
    duration = end_time - start_time
    with open(os.path.join(out_dir, 'scoring_log.txt'), 'a') as f:
        f.write(f"\nTotal run time: {duration.total_seconds():.2f} seconds\n")

    # print(f"\nRun complete. type_compare_Results saved in {out_dir}")


if __name__ == '__main__':
    main()
