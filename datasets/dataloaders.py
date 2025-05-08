import torch
from torch.utils.data import DataLoader, Subset  # Subset 用于从一个大的 Dataset 中创建子集视图
import numpy as np

# 从同级目录导入必要的模块
from .data_processor import load_and_preprocess_data
from .few_shot_dataset import FewShotDataset


def get_dataloaders(config: object) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    核心函数：负责加载原始数据、进行预处理、创建 FewShotDataset 实例，
    并最终生成用于训练、验证和测试的 PyTorch DataLoader。

    Args:
        config (object): 一个包含所有配置参数的对象或类实例。
                         期望它包含如 `config.data` (数据路径, seq_len等),
                         `config.fsl_task` (n_way, k_shot, n_query),
                         `config.training` (batch_size) 等属性。

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
            返回一个包含三个 DataLoader 的元组: (train_loader, val_loader, test_loader)。
            同时，此函数会动态地将从数据中推断出的 `feature_dim` 更新到传入的 `config.encoder` 对象中。
    """
    print("进入 get_dataloaders 函数...")
    # 1. 加载并预处理原始数据点
    #    `features` 是标准化后的数值数据，`labels` 是对应的0/1标签。
    print("步骤 1: 调用 load_and_preprocess_data 加载和预处理数据...")
    features, labels, feature_cols, scaler_instance = load_and_preprocess_data(
        config.data.incident_path,
        config.data.before_path,
        config.data.after_path
    )
    print(f"原始数据加载和预处理完成。特征形状: {features.shape}, 标签形状: {labels.shape}")

    # 动态确定特征维度并更新到配置对象中，供后续编码器初始化使用
    # features.shape[1] 即为特征的数量（维度）
    if hasattr(config.encoder, 'feature_dim'):  # 确保 encoder 配置对象有 feature_dim 属性
        config.encoder.feature_dim = features.shape[1]
        print(f"已将动态确定的特征维度 {config.encoder.feature_dim} 更新到配置中。")
    else:
        print("警告: 配置对象中 config.encoder 没有 feature_dim 属性，无法动态更新。编码器可能无法正确初始化。")

    # 2. 创建一个 FewShotDataset 实例 (我们称之为“全量任务采样器”)
    #    这个实例的 `__len__` 方法返回一个较大的值 (例如 `total_tasks_available_for_sampling`)，
    #    代表它理论上可以采样出非常多不同的小样本任务。
    #    我们将从这个“任务池”的“索引”中划分出训练、验证和测试集。
    #    这种做法的背景是：`FewShotDataset.__getitem__` 每次都随机生成任务，
    #    所以我们不是划分数据本身，而是划分“调用__getitem__的次数”或者说“任务索引”。

    # 定义 FewShotDataset 对象能够生成的任务总数（名义上的）。
    # 这个值应远大于 train/val/test Loader 在一个epoch中实际需要的任务数之和，
    # 以保证通过 Subset 划分索引时，各 Subset 仍能触发足够多次的、尽可能独特的任务采样。
    # 例如，如果训练2000个任务/epoch，验证500，测试500，那么这个值可以设为 10000, 20000 等。
    # 我们在配置文件中没有显式设置这个值，这里使用一个默认值。
    # 如果需要更精细控制，可以将其加入配置文件。
    total_tasks_available_for_sampling_pool_size = 20000  # 示例值
    print(f"步骤 2: 创建 FewShotDataset 实例 (任务采样池大小: {total_tasks_available_for_sampling_pool_size})...")

    full_dataset_for_task_sampling = FewShotDataset(
        features=features,
        labels=labels,
        seq_len=config.data.seq_len,
        n_way=config.fsl_task.n_way,
        k_shot=config.fsl_task.k_shot,
        n_query=config.fsl_task.n_query,
        num_tasks_per_epoch=total_tasks_available_for_sampling_pool_size  # 这是Dataset对象自身的“长度”
    )
    print("FewShotDataset 实例创建完成。")

    # 3. 根据配置中的划分比例，确定训练、验证、测试 DataLoader 各自需要多少个“任务索引”
    #    这些数量定义了每个 DataLoader 在一个“概念性 epoch”中会迭代多少次。
    #    例如，train_loader 的“长度”是 num_train_tasks。
    print("步骤 3: 计算训练、验证、测试集各自的任务索引数量...")
    train_ratio = config.data.train_split_ratio
    val_ratio = config.data.val_split_ratio
    test_ratio = 1.0 - train_ratio - val_ratio

    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1 and (
            train_ratio + val_ratio + test_ratio == 1.0)):
        # 如果比例不合理，进行调整或报错
        print(
            f"警告: 提供的划分比例不合理 (train: {train_ratio}, val: {val_ratio}, test: {test_ratio})。将尝试默认划分。")
        # 尝试一个默认的合理划分，例如 70/15/15
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        # 或者，更严格地抛出错误，要求用户修正配置
        # raise ValueError("训练、验证、测试集的划分比例总和必须为1，且每个比例都在 (0,1) 区间。")

    num_train_tasks = int(total_tasks_available_for_sampling_pool_size * train_ratio)
    num_val_tasks = int(total_tasks_available_for_sampling_pool_size * val_ratio)
    # 确保测试任务数至少为1，即使总数很少或比例计算后为0
    num_test_tasks = total_tasks_available_for_sampling_pool_size - num_train_tasks - num_val_tasks
    if num_test_tasks <= 0:  # 如果计算后测试任务数为0或负数
        print(f"警告: 计算得到的测试任务数为 {num_test_tasks}。至少需要1个测试任务。调整中...")
        # 从验证集中匀出一些给测试集，或直接分配一个最小数量
        if num_val_tasks > 100:  # 假设验证集至少有100个任务可以匀
            num_test_tasks = max(1, int(num_val_tasks * 0.2))  # 例如匀出20%
            num_val_tasks -= num_test_tasks
        else:  # 如果验证集也很少，就尽量保证测试集有几个任务
            num_test_tasks = max(1, int(total_tasks_available_for_sampling_pool_size * 0.05))  # 至少5%给测试
            # 可能需要重新调整 num_train_tasks 和 num_val_tasks 以确保总和不变，这里简化处理

    print(f"计划的训练任务索引数: {num_train_tasks}")
    print(f"计划的验证任务索引数: {num_val_tasks}")
    print(f"计划的测试任务索引数: {num_test_tasks}")

    # 4. 创建一个从 0 到 total_tasks_available_for_sampling_pool_size-1 的索引数组，并打乱它
    #    这些索引将用于从 full_dataset_for_task_sampling 中通过 Subset 来“抽取”任务。
    print("步骤 4: 创建并打乱任务索引池...")
    all_task_indices = np.arange(total_tasks_available_for_sampling_pool_size)
    np.random.seed(config.random_seed)  # 确保索引划分的可复现性
    np.random.shuffle(all_task_indices)
    print("任务索引池打乱完成。")

    # 5. 根据计算出的任务数量，从打乱后的索引数组中分配索引给训练、验证和测试集
    train_indices = all_task_indices[:num_train_tasks]
    val_indices = all_task_indices[num_train_tasks: num_train_tasks + num_val_tasks]
    # 确保测试集索引不越界，并使用计算好的 num_test_tasks
    test_indices = all_task_indices[num_train_tasks + num_val_tasks: num_train_tasks + num_val_tasks + num_test_tasks]

    # 6. 使用 torch.utils.data.Subset 类创建特定于训练、验证、测试的“数据集视图”
    #    Subset 包装了原始的 full_dataset_for_task_sampling，但只暴露通过 train/val/test_indices 指定的“任务”。
    #    当 DataLoader 从这些 Subset 中请求数据时，Subset 会使用其分配到的索引
    #    去调用 full_dataset_for_task_sampling 的 __getitem__ 方法来动态生成一个小样本任务。
    print("步骤 5: 创建训练、验证、测试的 Subset 实例...")
    train_subset = Subset(full_dataset_for_task_sampling, train_indices)
    val_subset = Subset(full_dataset_for_task_sampling, val_indices)
    test_subset = Subset(full_dataset_for_task_sampling, test_indices)

    print(f"训练 Subset 创建完成，包含任务索引数: {len(train_subset)}")
    print(f"验证 Subset 创建完成，包含任务索引数: {len(val_subset)}")
    print(f"测试 Subset 创建完成，包含任务索引数: {len(test_subset)}")
    if len(test_subset) < config.evaluation.num_test_episodes:
        print(
            f"警告: 测试 Subset ({len(test_subset)}) 的长度小于配置中要求的评估任务数 ({config.evaluation.num_test_episodes})。评估时将以 Subset 长度为准。")

    # 7. 创建 DataLoader 实例
    #    DataLoader 负责从 Subset 中按批次 (batch_size) 加载任务，并可以进行打乱、多进程加载等。
    print("步骤 6: 创建 DataLoader 实例...")
    # 确定是否在 GPU 上运行，如果是，可以启用 pin_memory 来加速数据从 CPU 到 GPU 的传输
    pin_memory_flag = torch.cuda.is_available() if config.device.lower() != 'cpu' else False

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=config.training.batch_size,  # 每个批次包含多少个小样本任务
        shuffle=True,  # 在每个 epoch 开始时打乱任务的顺序，有助于模型学习
        num_workers=0,  # 使用多少个子进程加载数据。0 表示在主进程中加载。
        # 在 Windows 上或简单的数据集上，0 通常更稳定。可根据CPU核心数调整。
        pin_memory=pin_memory_flag
    )
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=config.training.batch_size,  # 验证时批次大小可以与训练时相同，或根据内存情况调整
        shuffle=False,  # 验证和测试时通常不需要打乱数据顺序
        num_workers=0,
        pin_memory=pin_memory_flag
    )
    test_loader = DataLoader(
        dataset=test_subset,
        batch_size=config.training.batch_size,  # 测试时批次大小，用于最终评估
        # 有时会设为1，逐个任务评估，但如果模型能处理批次则可以保持与验证一致
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory_flag
    )
    print("DataLoader 实例创建完成。")
    print("get_dataloaders 函数执行完毕。")

    return train_loader, val_loader, test_loader