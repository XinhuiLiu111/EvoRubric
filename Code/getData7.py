import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


class GetData:
    def __init__(self, dataset, essay_set_value):
        self.dataset = dataset
        self.essay_set_value = essay_set_value
        # 获取项目根目录路径
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_prompt(self):
        # 根据essay_set_value返回相应的prompt
        file_path = os.path.join(self.project_root, 'Data', self.dataset, 'prompts', f'Essaay Set #{self.essay_set_value}.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt

    def get_rubric(self):
        # 根据essay_set_value返回相应的rubric
        file_path = os.path.join(self.project_root, 'Data', self.dataset, 'scoring_rubric', f'Essaay Set #{self.essay_set_value}.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            rubric = file.read()
        return rubric

    def get_score_range(self):
        # 根据essay_set_value返回相应的score_range
        file_path = os.path.join(self.project_root, 'Data', self.dataset, 'score range.xlsx')
        df = pd.read_excel(file_path)

        # 确保essay_set列为整数类型
        df['essay_set'] = df['essay_set'].astype(int)

        # 获取score_range并解析为元组
        score_range_str = df[df['essay_set'] == self.essay_set_value]['score_range'].values[0]

        try:
            # 如果是形如 "2-12 points" 的格式
            if isinstance(score_range_str, str) and 'points' in score_range_str:
                # 移除 "points" 并分割数字
                numbers = score_range_str.replace('points', '').strip().split('-')
                return (int(numbers[0]), int(numbers[1]))

            # 如果是字符串格式如"(0,5)"
            elif isinstance(score_range_str, str):
                score_range_str = score_range_str.strip('()[]{}').split(',')
                return (int(score_range_str[0]), int(score_range_str[1]))

            # 如果已经是元组或列表格式
            elif isinstance(score_range_str, (tuple, list)):
                return (int(score_range_str[0]), int(score_range_str[1]))

            else:
                raise ValueError(f"Unexpected score_range format: {score_range_str}")

        except Exception as e:
            print(f"Error parsing score range: {score_range_str}")
            print(f"Error details: {str(e)}")
            raise

    def calculate_possible_batches(self, train_data, score_col='domain1_score'):
        """
        计算训练集可以分成多少个batch

        参数:
        train_data: DataFrame, 训练集数据
        score_col: str, 分数列名

        返回:
        int, 可能的batch数量
        """
        # 为训练集添加分数等级
        min_score = train_data[score_col].min()
        max_score = train_data[score_col].max()
        score_range = max_score - min_score
        interval = score_range / 2

        def get_score_level(score):
            if score <= min_score + interval:
                return 'low'
            else:
                return 'high'

        train_data = train_data.copy()
        train_data['score_level'] = train_data[score_col].apply(get_score_level)

        # 计算总样本数
        total_samples = len(train_data)

        # 每个batch需要2个样本，计算可能的batch数量
        possible_batches = total_samples // 2

        # 打印各个等级的样本分布情况（用于参考）
        level_counts = train_data['score_level'].value_counts()
        # print("各等级样本分布：")
        # print(level_counts)

        return possible_batches

    def create_score_levels(self, df, score_col='domain1_score'):
        """
        为数据集添加分数等级，根据分数实际范围划分为2个等级

        参数:
        df: DataFrame, 包含分数的数据集
        score_col: str, 分数列名

        返回:
        DataFrame: 添加了score_level列的数据集
        """
        df = df.copy()
        min_score = df[score_col].min()
        max_score = df[score_col].max()
        score_range = max_score - min_score
        interval = score_range / 2

        def get_score_level(score):
            if score <= min_score + interval:
                return 'low'
            else:
                return 'high'

        df['score_level'] = df[score_col].apply(get_score_level)
        return df

    def create_batch(self, data, batch_size=2):
        """
        从当前数据中创建一个batch

        参数:
        data: DataFrame, 当前可用的训练数据
        batch_size: int, 期望的batch大小

        返回:
        tuple: (batch样本, 剩余样本)
        """
        if len(data) == 0:
            return None, None

        # 为当前数据添加分数等级
        leveled_data = self.create_score_levels(data)

        # 获取当前的等级分布
        level_counts = leveled_data['score_level'].value_counts()
        batch_size = min(batch_size, len(level_counts))

        # 随机抽取样本
        selected_samples = []
        remaining_data = leveled_data.copy()
        selected_levels = []

        # 尝试获取batch_size个样本
        while len(selected_samples) < batch_size and not remaining_data.empty:
            # 获取当前剩余数据的等级分布
            current_levels = remaining_data['score_level'].unique()

            if len(current_levels) == 0:
                break

            # 随机选择一个等级
            remaining_levels = list(set(current_levels) - set(selected_levels))
            if remaining_levels:
                selected_level = np.random.choice(np.array(remaining_levels))
                selected_levels.append(selected_level)
            else:
                return None, None

            # 从选中的等级中随机抽取一个样本
            level_samples = remaining_data[remaining_data['score_level'] == selected_level]
            if not level_samples.empty:
                selected_sample = level_samples.sample(n=1, random_state=42)
                selected_samples.append(selected_sample)

                # 从剩余数据中移除已选样本
                remaining_data = remaining_data.drop(selected_sample.index)

        if selected_samples:
            batch = pd.concat(selected_samples)
            # 移除score_level列
            batch = batch.drop('score_level', axis=1)
            remaining_data = remaining_data.drop('score_level', axis=1)
            return batch, remaining_data
        return None, None

    def get_dataset(self, file_path):
        # 读取Excel数据集文件
        df = pd.read_excel(file_path)

        # 删除domain1_score列包含缺失值的行，并转化为int
        df = df.dropna(subset=['domain1_score'])
        df['domain1_score'] = df['domain1_score'].astype(int)
        df['essay_set'] = df['essay_set'].astype(int)

        # 提取所需的列
        df = df[['essay_id', 'essay_set', 'essay', 'domain1_score']]

        # 根据输入的essay_set值进行筛选
        df = df[df['essay_set'] == self.essay_set_value].copy()

        # 添加这一行：限制数据量为200条
        df = df.sample(n=min(200, len(df)), random_state=42)
        # 打印这些数据的分数分布
        print(f"\n随机选取的{len(df)}条作文的分数分布:")
        print(df['domain1_score'].value_counts().sort_index())

        # 获取分数列名
        score_col = 'domain1_score'

        # 获取唯一分数值及其样本数
        score_counts = df[score_col].value_counts()

        # 处理小样本分数段
        min_samples_per_stratum = 3  # 每层最少样本数
        valid_scores = score_counts[score_counts >= min_samples_per_stratum].index

        # 只保留样本数足够的分数段
        df = df[df[score_col].isin(valid_scores)]

        # 检查每个分数等级的样本数
        score_counts = df['domain1_score'].value_counts()
        min_samples = score_counts.min()
        
        # 如果最小样本数大于等于2，使用stratify，否则不使用
        stratify = df['domain1_score'] if min_samples >= 2 else None
        
        # 第一步：将数据集分为训练集(120篇)和剩余部分(80篇)
        train_data, temp_data = train_test_split(
            df,
            train_size=120,  # 修改为120篇
            stratify=None,   # 放弃分层抽样，改用随机抽样
            random_state=42  # 保持固定的随机种子
        )

        # 第二步：将剩余的80篇平均分为验证集(40篇)和测试集(40篇)
        valid_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,  # 各40篇
            stratify=None,  # 放弃分层抽样，改用随机抽样
            random_state=42 # 保持固定的随机种子
        )

        # 打印数据集大小和分布
        print(f"\n数据集划分:")
        print(f"训练集大小: {len(train_data)} 篇")
        print(f"验证集大小: {len(valid_data)} 篇")
        print(f"测试集大小: {len(test_data)} 篇")

        return train_data, valid_data, test_data


if __name__ == "__main__":
    # 使用函数并传入指定的数据集和essay_set值
    dataset = 'ASAP'
    essay_set_value = 8  # 文章所属集合
    # 获取项目根目录路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(project_root, 'Data', 'ASAP', 'training_set_rel3.xlsx')

    data = GetData(dataset, essay_set_value)
    prompt = data.get_prompt()
    score_range = data.get_score_range()
    train_set, val_set, test_set = data.get_dataset(file_path)


#输出训练集、测试集、验证集的大小
    # # 创建保存目录
    # save_dir = os.path.join('data_splits', f'essay_set_{essay_set_value}')
    # os.makedirs(save_dir, exist_ok=True)
    #
    # # 将train_set从batches列表转换为DataFrame
    # if train_set:  # 确保train_set不为空
    #     train_df = pd.concat(train_set) if isinstance(train_set, list) else train_set
    #     train_df.to_excel(os.path.join(save_dir, 'train_set.xlsx'), index=False)
    #
    # # 保存验证集
    # val_set.to_excel(os.path.join(save_dir, 'validation_set.xlsx'), index=False)
    #
    # # 保存测试集
    # test_set.to_excel(os.path.join(save_dir, 'test_set.xlsx'), index=False)
    #
    # print(f"\n数据集已保存至: {save_dir}")
    # print(f"训练集大小: {len(train_df) if train_set else 0}")
    # print(f"验证集大小: {len(val_set)}")
    # print(f"测试集大小: {len(test_set)}")

    # 打印获取的数据
    print("\nPrompt:")
    print(prompt)
    print("\nScore Range:")
    print(score_range)
