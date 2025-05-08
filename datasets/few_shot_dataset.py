import torch
from torch.utils.data import Dataset  # PyTorch Dataset 基类
import numpy as np


class FewShotDataset(Dataset):
    """
    为小样本学习 (Few-Shot Learning, FSL) 任务动态生成数据样本。

    这个类的核心功能是，在每次被 DataLoader 调用 `__getitem__` 时，
    它会从提供的全部预处理数据（`features` 和 `labels`）中，
    根据 N-way, K-shot, Q-query 的设定，随机采样并构建一个“小样本任务” (episode)。
    一个小样本任务包含一个支持集 (support set) 和一个查询集 (query set)。
    """

    def __init__(self,
                 features: np.ndarray,  # 预处理后的整体特征数据
                 labels: np.ndarray,  # 对应的标签
                 seq_len: int,  # 每个时间序列样本的长度
                 n_way: int,  # N-way: 任务中的类别数
                 k_shot: int,  # K-shot: 每类在支持集中的样本数
                 n_query: int,  # N-query: 每类在查询集中的样本数
                 num_tasks_per_epoch: int = 10000  # 定义数据集的“名义长度”
                 ):
        """
        初始化 FewShotDataset。

        Args:
            features (np.ndarray): 形状为 `[总时间点数, 特征维度]` 的特征数组。
                                   这是经过 `load_and_preprocess_data` 处理后的全部数据。
            labels (np.ndarray): 形状为 `[总时间点数]` 的标签数组。
                                  与 `features` 中的每个时间点对应。
            seq_len (int): 将连续的原始时间点数据构造成时间序列片段时的长度。
                           例如，`seq_len=24` 表示每个序列样本包含24个连续的时间点。
            n_way (int): 每个小样本任务中包含的类别数量。
                         对于二分类（如正常/异常），`n_way=2`。
            k_shot (int): 在每个任务的支持集中，每个类别提供的样本（序列片段）数量。
            n_query (int): 在每个任务的查询集中，每个类别提供的样本（序列片段）数量。
            num_tasks_per_epoch (int, optional):
                `__len__` 方法将返回此值。它定义了 DataLoader 在一个 epoch 中
                名义上可以从这个数据集中采样多少个任务。这并不意味着数据集中只有
                这么多唯一的任务组合，而更像是一个迭代次数的上限或期望值。
                默认为 10000。
        """
        self.features = features
        self.labels = labels
        self.seq_len = seq_len
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.num_tasks_per_epoch = num_tasks_per_epoch

        # 1. 按类别分离原始数据点的索引
        #    这样可以方便地从特定类别中采样数据来构建序列。
        self.normal_indices = np.where(labels == 0)[0]  # 获取所有标签为0（正常）的样本在 `features` 中的行索引
        self.anomaly_indices = np.where(labels == 1)[0]  # 获取所有标签为1（异常）的样本在 `features` 中的行索引

        print(f"FewShotDataset 初始化: 原始正常样本点数 (label 0): {len(self.normal_indices)}")
        print(f"FewShotDataset 初始化: 原始异常样本点数 (label 1): {len(self.anomaly_indices)}")

        # 2. 将连续的同类数据点构造成时间序列片段
        #    `_create_sequences_for_class` 会将例如属于“正常”类的所有连续时间点，
        #    按照 `seq_len` 切割成多个不重叠的序列片段。
        self.normal_sequences = self._create_sequences_for_class(self.normal_indices)
        self.anomaly_sequences = self._create_sequences_for_class(self.anomaly_indices)

        print(f"FewShotDataset 初始化: 基于 seq_len={seq_len} 创建的正常序列片段数: {len(self.normal_sequences)}")
        print(f"FewShotDataset 初始化: 基于 seq_len={seq_len} 创建的异常序列片段数: {len(self.anomaly_sequences)}")

        # 3. 健全性检查：确保每个类别有足够的序列片段来构建至少一个小样本任务
        #    每个类别至少需要 `k_shot` (支持集) + `n_query` (查询集) 个不同的序列片段。
        min_sequences_needed_per_class = self.k_shot + self.n_query
        if len(self.normal_sequences) < min_sequences_needed_per_class:
            raise ValueError(
                f"正常类别 (label 0) 的序列片段数量 ({len(self.normal_sequences)}) "
                f"不足以满足一个任务的需求 ({min_sequences_needed_per_class} = "
                f"{self.k_shot} support + {self.n_query} query)。请检查数据量或 seq_len 设置。"
            )
        if len(self.anomaly_sequences) < min_sequences_needed_per_class:
            raise ValueError(
                f"异常类别 (label 1) 的序列片段数量 ({len(self.anomaly_sequences)}) "
                f"不足以满足一个任务的需求 ({min_sequences_needed_per_class} = "
                f"{self.k_shot} support + {self.n_query} query)。请检查数据量或 seq_len 设置。"
            )
        print("FewShotDataset 初始化完成，序列数量检查通过。")

    def _create_sequences_for_class(self, class_indices: np.ndarray) -> list:
        """
        辅助函数：为特定类别的所有样本点创建不重叠的时间序列片段。

        Args:
            class_indices (np.ndarray): 一个 NumPy 数组，包含属于某个特定类别的所有样本点
                                        在原始 `self.features` 数组中的行索引。

        Returns:
            list[np.ndarray]: 一个列表，其中每个元素是一个 NumPy 数组，代表一个时间序列片段。
                              每个片段的形状为 `[self.seq_len, feature_dimension]`。
        """
        sequences = []  # 用于存储生成的序列片段
        num_class_points = len(class_indices)  # 该类别总共有多少个时间点

        # 以 self.seq_len 为步长，遍历该类别的所有时间点索引
        for i in range(num_class_points // self.seq_len):
            # 计算当前序列片段在 class_indices 中的起始和结束位置
            start_index_in_class_indices_array = i * self.seq_len
            end_index_in_class_indices_array = start_index_in_class_indices_array + self.seq_len

            # 确保切片不会越界 (虽然 // self.seq_len 已经保证了大部分情况，但作为双重检查)
            if end_index_in_class_indices_array <= num_class_points:
                # 从 class_indices 中获取这部分对应的原始特征数据索引
                actual_feature_indices = class_indices[
                                         start_index_in_class_indices_array:end_index_in_class_indices_array]
                # 使用这些真实索引从 self.features 中提取对应的序列片段数据
                sequence_data = self.features[actual_feature_indices]
                sequences.append(sequence_data)
        return sequences

    def __len__(self) -> int:
        """
        返回数据集的“长度”。这个长度通常用于告知 DataLoader 在一个 epoch 中
        期望进行多少次 `__getitem__` 调用（即生成多少个任务）。
        """
        return self.num_tasks_per_epoch

    def __getitem__(self, idx: int) -> tuple:
        """
        生成并返回一个小样本任务 (episode)。
        参数 `idx` 在此实现中通常不被直接使用，因为每次调用都是随机采样以保证任务的多样性。

        Returns:
            tuple: 包含四个元素的元组 `(support_x, support_y, query_x, query_y)`
                - `support_x` (torch.FloatTensor): 支持集的特征数据。
                    形状: `[n_way * k_shot, seq_len, feature_dim]`
                - `support_y` (torch.LongTensor): 支持集的标签。
                    形状: `[n_way * k_shot]`
                - `query_x` (torch.FloatTensor): 查询集的特征数据。
                    形状: `[n_way * n_query, seq_len, feature_dim]`
                - `query_y` (torch.LongTensor): 查询集的标签。
                    形状: `[n_way * n_query]`
        """
        # 当前实现主要针对二分类（正常/异常）任务
        if self.n_way != 2:
            # 如果要支持更通用的 N-way 分类，这里的类别采样逻辑需要修改。
            # 例如，需要从所有可用类别中随机选择 N 个类别，然后再为这 N 个类别采样 K-shot 和 Q-query。
            raise NotImplementedError("当前 FewShotDataset 实现仅硬编码支持 2-way 分类 (正常/异常)。")

        # 初始化用于存储当前任务数据的列表
        support_x_task_list = []
        support_y_task_list = []
        query_x_task_list = []
        query_y_task_list = []

        # --- 为当前任务采样正常类别 (标签 0) 的数据 ---
        num_sequences_to_sample_for_normal = self.k_shot + self.n_query
        # 从所有已创建的正常序列中，不重复地随机选择 `num_sequences_to_sample_for_normal` 个序列的索引。
        # 由于在 __init__ 中已经检查过序列数量足够，所以 `replace=False` 是安全的。
        chosen_normal_sequence_indices = np.random.choice(
            a=len(self.normal_sequences),  # 从0到len-1的索引中选择
            size=num_sequences_to_sample_for_normal,
            replace=False  # 不重复采样
        )

        for i, chosen_seq_idx in enumerate(chosen_normal_sequence_indices):
            sequence_data = self.normal_sequences[chosen_seq_idx]  # 获取实际的序列数据
            if i < self.k_shot:  # 前 k_shot 个样本作为支持集
                support_x_task_list.append(sequence_data)
                support_y_task_list.append(0)  # 正常类别的标签为 0
            else:  # 剩余的 n_query 个样本作为查询集
                query_x_task_list.append(sequence_data)
                query_y_task_list.append(0)

                # --- 为当前任务采样异常类别 (标签 1) 的数据 ---
        num_sequences_to_sample_for_anomaly = self.k_shot + self.n_query
        chosen_anomaly_sequence_indices = np.random.choice(
            a=len(self.anomaly_sequences),
            size=num_sequences_to_sample_for_anomaly,
            replace=False
        )

        for i, chosen_seq_idx in enumerate(chosen_anomaly_sequence_indices):
            sequence_data = self.anomaly_sequences[chosen_seq_idx]
            if i < self.k_shot:
                support_x_task_list.append(sequence_data)
                support_y_task_list.append(1)  # 异常类别的标签为 1
            else:
                query_x_task_list.append(sequence_data)
                query_y_task_list.append(1)

        # --- 将列表转换为 PyTorch Tensors ---
        # `np.array()` 会将包含多个 [seq_len, feature_dim] 形状数组的列表，
        # 转换成一个 [num_total_samples_in_set, seq_len, feature_dim] 的 NumPy 数组。
        # 例如, support_x_task_list 包含 (n_way * k_shot) 个序列。

        # DataLoader 会将这些单个任务的输出打包成一个批次 (batch of tasks)。
        # 因此，FSL 方法模型 (如 PrototypicalNetwork) 的 `forward` 方法需要能够处理这种批次化的任务数据。

        return (
            torch.FloatTensor(np.array(support_x_task_list)),  # 支持集特征
            torch.LongTensor(np.array(support_y_task_list)),  # 支持集标签
            torch.FloatTensor(np.array(query_x_task_list)),  # 查询集特征
            torch.LongTensor(np.array(query_y_task_list))  # 查询集标签
        )