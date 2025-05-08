import torch
import torch.nn as nn
from .base_encoder import BaseEncoder  # 从同级目录的 base_encoder 模块导入 BaseEncoder 类


class TimeSeriesEncoder(BaseEncoder):
    """
    基于 Transformer 的时间序列编码器实现。
    它继承自 `BaseEncoder`，并将输入的时间序列数据编码为固定长度的嵌入向量。
    这个编码器是小样本学习方法中用于提取特征的核心组件之一。
    """

    def __init__(self,
                 feature_dim: int,  # 输入时间序列的原始特征维度
                 hidden_dim: int = 128,  # Transformer 模型的内部隐藏维度 (d_model)
                 num_layers: int = 4,  # Transformer 编码器中 EncoderLayer 的层数
                 nhead: int = 8,  # Transformer 多头注意力机制中的头数
                 dropout: float = 0.1  # 在 Transformer 各层中使用的 Dropout 比率
                 ):
        """
        初始化 TimeSeriesEncoder。

        Args:
            feature_dim (int): 输入时间序列中每个时间点的特征数量。
                               例如，如果每个时间点有10个监控指标，则 feature_dim=10。
            hidden_dim (int, optional): Transformer 模型的内部工作维度，也称为 d_model。
                                        它决定了模型参数量和表达能力。默认为 128。
            num_layers (int, optional): Transformer 编码器堆叠的 EncoderLayer 的数量。
                                       层数越多，模型能学习到更复杂的模式，但也更难训练。默认为 4。
            nhead (int, optional): 多头自注意力机制中的“头”的数量。
                                  `hidden_dim` 必须能被 `nhead` 整除。默认为 8。
            dropout (float, optional): 在 Transformer 的多个子层（如自注意力、前馈网络）之后应用的 Dropout 比率。
                                      用于正则化，防止过拟合。默认为 0.1。
        """
        super(TimeSeriesEncoder, self).__init__()  # 调用父类 BaseEncoder 的构造函数

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 1. 输入投影层 (Input Projection Layer)
        # 将原始的 `feature_dim` 维度的输入特征，通过一个线性变换映射到 Transformer 所需的 `hidden_dim` 维度。
        self.input_projection = nn.Linear(feature_dim, hidden_dim)

        # 2. 位置编码层 (Positional Encoding Layers) - 简化的自定义版本
        # Transformer 本身不包含序列位置信息，需要额外添加。
        # 这里的实现是一种自定义的、通过一系列可学习的线性层来注入位置信息的方式。
        # 标准的 Transformer 通常使用固定的正弦/余弦位置编码或可学习的 nn.Embedding 作为位置编码。
        # 这个自定义版本可能需要实验验证其有效性，或者可以替换为标准实现。
        self.pos_encoder = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)  # 创建3个线性层
        ])

        # 3. Transformer 编码器层 (Transformer Encoder Layer)
        # PyTorch 提供的标准 TransformerEncoderLayer 模块，它包含一个多头自注意力机制和一个前馈神经网络。
        # `batch_first=True` 表示输入和输出张量的形状将是 (batch_size, sequence_length, feature_dimension)。
        encoder_layer_config = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # 模型维度，必须等于 input_projection 的输出维度
            nhead=nhead,  # 多头注意力头数
            dim_feedforward=hidden_dim * 4,  # 前馈网络内部的维度，通常是 d_model 的2到4倍
            dropout=dropout,  # Dropout 比率
            batch_first=True  # 重要！确保输入输出格式为 [批次, 序列, 特征]
        )
        # 4. 完整的 Transformer 编码器 (Transformer Encoder)
        # PyTorch 提供的标准 TransformerEncoder 模块，它将多个 TransformerEncoderLayer 堆叠起来。
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer_config,  # 使用上面定义的单层配置
            num_layers=num_layers  # 堆叠的层数
        )

        # 5. 自适应层 (Adaptive Layer)
        # 在 Transformer 编码器输出之后添加的一个小型前馈网络，用于进一步处理序列的整体表示。
        # 这里的结构是 隐藏 -> 隐藏/2 -> 隐藏，并使用了 GELU 激活函数。
        self.adaptive_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),  # GELU 是一种平滑的激活函数，常用于 Transformer 模型
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        print(
            f"TimeSeriesEncoder 初始化完成: feature_dim={feature_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, nhead={nhead}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TimeSeriesEncoder 的前向传播逻辑。

        Args:
            x (torch.Tensor): 输入的时间序列批次数据。
                              形状: `[N, S, F]`
                              N: 批次大小 (或者在小样本任务中，是支持集/查询集样本的总数)
                              S: 序列长度 (config.data.seq_len)
                              F: 每个时间点的原始特征维度 (config.encoder.feature_dim)

        Returns:
            torch.Tensor: 编码后的序列表示 (嵌入向量)。
                          形状: `[N, H]`
                          N: 与输入 N 相同
                          H: 编码器的隐藏维度 (config.encoder.hidden_dim)
        """
        # 1. 输入投影: (N, S, F) -> (N, S, H)
        #    将每个时间点的 F 维特征映射到 H 维。
        x_projected = self.input_projection(x)

        # 2. 自定义位置编码: 形状保持 (N, S, H)
        #    通过一系列线性变换和元素操作为序列中的每个位置添加信息。
        x_pos_encoded = x_projected
        for i, layer in enumerate(self.pos_encoder):
            if i % 2 == 0:  # 偶数层：残差连接式
                x_pos_encoded = x_pos_encoded + layer(x_pos_encoded)
            else:  # 奇数层：门控式（元素乘法配合 sigmoid）
                x_pos_encoded = x_pos_encoded * torch.sigmoid(layer(x_pos_encoded))

        # 3. Transformer 编码器处理: (N, S, H) -> (N, S, H)
        #    在加入了位置信息后，通过多层自注意力机制和前馈网络捕捉序列内部的依赖关系。
        transformer_output = self.transformer_encoder(x_pos_encoded)

        # 4. 序列聚合 (Sequence Aggregation): (N, S, H) -> (N, H)
        #    将 Transformer 输出的整个序列的表示（每个时间步都有一个 H 维向量）
        #    聚合成一个单一的 H 维向量来代表整个序列。
        #    这里使用的是对序列长度维度取平均值的方法。
        #    其他常见的聚合方法包括：
        #    - 取 [CLS] token 的输出 (如果模型设计中加入了类似 BERT 的 [CLS] token)
        #    - 最大池化 (Max Pooling) 或注意力池化 (Attention Pooling)
        sequence_embedding = torch.mean(transformer_output, dim=1)

        # 5. 自适应层处理: (N, H) -> (N, H)
        #    对聚合后的序列嵌入进行进一步的非线性变换。
        #    同样使用了残差连接，将自适应层的输出与其输入 (sequence_embedding) 相加。
        #    这有助于梯度的传播和模型的学习。
        final_embedding = sequence_embedding + self.adaptive_layer(sequence_embedding)

        return final_embedding