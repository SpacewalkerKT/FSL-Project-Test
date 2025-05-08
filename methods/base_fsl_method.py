import torch
import torch.nn as nn
from abc import ABC, abstractmethod  # 导入抽象基类相关工具
from encoders.base_encoder import BaseEncoder  # 导入编码器基类，确保路径正确


class BaseFSLMethod(nn.Module, ABC):
    """
    小样本学习 (Few-Shot Learning, FSL) 方法的抽象基类。

    这个基类为所有具体的小样本学习算法（如原型网络、MAML等）提供了一个统一的结构。
    它强调了 FSL 方法通常依赖于一个特征编码器来处理输入数据。

    主要职责：
    1. 接收一个编码器实例。
    2. 定义一个抽象的 `forward` 方法，该方法应由具体的 FSL 算法来实现，
       用于处理一个小样本任务（通常包含支持集和查询集）并输出预测。
    """

    def __init__(self, encoder: BaseEncoder):
        """
        初始化基础 FSL 方法。

        Args:
            encoder (BaseEncoder): 一个编码器模块的实例 (例如 TimeSeriesEncoder)。
                                   这个编码器将被 FSL 方法用来将输入数据（如时间序列片段）
                                   转换为低维的嵌入特征。
        """
        super(BaseFSLMethod, self).__init__()  # 调用 nn.Module 的构造函数
        self.encoder = encoder  # 保存传入的编码器实例

    @abstractmethod  # 声明这是一个抽象方法，子类必须实现它
    def forward(self,
                support_x_batch: torch.Tensor,
                support_y_batch: torch.Tensor,
                query_x_batch: torch.Tensor,
                n_way: int,
                k_shot: int,
                n_query: int
                ) -> torch.Tensor:
        """
        处理一批 (batch) 小样本任务的前向传播。
        这是 FSL 方法的核心，具体实现因算法而异。

        Args:
            support_x_batch (torch.Tensor):
                一个批次的支持集样本特征。
                形状通常为: `[batch_size, num_total_support_samples, seq_len, feature_dim]`
                其中 `num_total_support_samples = n_way * k_shot`。
                `batch_size` 是指元批次大小，即一次处理多少个独立的 FSL 任务。
            support_y_batch (torch.Tensor):
                一个批次的支持集样本标签。
                形状通常为: `[batch_size, num_total_support_samples]`。
            query_x_batch (torch.Tensor):
                一个批次的查询集样本特征。
                形状通常为: `[batch_size, num_total_query_samples, seq_len, feature_dim]`
                其中 `num_total_query_samples = n_way * n_query`。
            n_way (int): 当前任务中的类别数量 (N-way)。
            k_shot (int): 当前任务中每个类别的支持样本数量 (K-shot)。
            n_query (int): 当前任务中每个类别的查询样本数量。这个参数可能不总是在
                           `forward` 中直接使用，但传递它可以使方法更通用，
                           例如用于内部的形状断言或逻辑分支。

        Returns:
            torch.Tensor:
                模型对批次中所有查询集样本的预测 logits (未经过 softmax 的原始输出)。
                形状通常为: `[batch_size * num_total_query_samples, n_way]`。
                这个输出将用于计算损失函数 (例如交叉熵损失)。
        """
        pass  # 子类必须提供这个方法的具体实现

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        一个辅助函数，用于方便地调用内部编码器来提取输入 `x` 的特征嵌入。

        Args:
            x (torch.Tensor): 需要进行编码的输入数据。
                              通常期望的形状是 `[num_samples, seq_len, feature_dim]`，
                              其中 `num_samples` 可以是任意数量的样本 (例如，一个任务中所有的支持集和查询集样本展平后的总数)。

        Returns:
            torch.Tensor: 通过编码器 `self.encoder` 得到的特征嵌入。
                          形状通常是 `[num_samples, embed_dim]`。
        """
        # 直接调用存储的 self.encoder 实例的 forward 方法
        return self.encoder(x)