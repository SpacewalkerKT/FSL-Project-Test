import torch
import torch.nn.functional as F  # PyTorch 的函数模块，常用于损失函数、激活函数等
from .base_fsl_method import BaseFSLMethod  # 从同级目录导入 BaseFSLMethod 基类
from encoders.base_encoder import BaseEncoder  # 导入编码器基类 (尽管构造函数中类型提示已用)


class PrototypicalNetwork(BaseFSLMethod):
    """
    原型网络 (Prototypical Network) 的具体实现。
    它是一种基于度量学习的小样本学习方法。

    核心思想：
    1. 使用编码器将支持集和查询集的样本映射到同一个嵌入空间。
    2. 对于支持集中的每个类别，计算其所有样本嵌入的均值，这个均值向量作为该类别的“原型”(prototype)。
    3. 对于查询集中的每个样本，计算其嵌入与所有类别原型的距离（或相似度）。
    4. 基于距离（或相似度）进行分类，通常将查询样本归类到距离其最近的原型所属的类别。
    """

    def __init__(self, encoder: BaseEncoder):
        """
        初始化原型网络。

        Args:
            encoder (BaseEncoder): 一个编码器模块的实例，用于将输入数据转换为特征嵌入。
        """
        super(PrototypicalNetwork, self).__init__(encoder)  # 调用父类 BaseFSLMethod 的构造函数，传入编码器
        print("PrototypicalNetwork 方法已初始化。")

    def forward(self,
                support_x_batch: torch.Tensor,
                support_y_batch: torch.Tensor,
                query_x_batch: torch.Tensor,
                n_way: int,
                k_shot: int,
                n_query: int
                ) -> torch.Tensor:
        """
        原型网络的前向传播，处理一批 (batch_size) 小样本任务。

        Args:
            support_x_batch (torch.Tensor):
                支持集样本的特征数据。
                形状: `[batch_size, num_support_per_task, seq_len, feature_dim]`
                其中 `num_support_per_task = n_way * k_shot`。
            support_y_batch (torch.Tensor):
                支持集样本的标签。
                形状: `[batch_size, num_support_per_task]`。
            query_x_batch (torch.Tensor):
                查询集样本的特征数据。
                形状: `[batch_size, num_query_per_task, seq_len, feature_dim]`
                其中 `num_query_per_task = n_way * n_query`。
            n_way (int): 每个任务中的类别数量。
            k_shot (int): 每个类别在支持集中的样本数量。
            n_query (int): 每个类别在查询集中的样本数量。

        Returns:
            torch.Tensor:
                模型对所有查询集样本的预测 logits。
                形状: `[batch_size * num_query_per_task, n_way]`。
                例如，如果 batch_size=4, n_way=2, n_query=5，则 num_query_per_task=10，
                输出形状为 `[4 * 10, 2] = [40, 2]`。
        """
        batch_size = support_x_batch.size(0)  # 获取元批次大小，即同时处理多少个 FSL 任务

        # 计算每个任务中支持样本和查询样本的总数
        num_support_per_task = n_way * k_shot
        num_query_per_task = n_way * n_query

        # 从输入张量中获取序列长度和原始特征维度 (主要用于 reshape)
        seq_len = support_x_batch.size(-2)
        feature_dim = support_x_batch.size(-1)

        # 1. 特征编码 (Embedding)
        #    首先，将所有任务中的所有支持集样本和查询集样本“拉平”，
        #    以便一次性通过编码器进行特征提取。
        #    例如，support_x_batch 从 [B, NK, S, F] 变为 [B*NK, S, F]
        support_x_flat = support_x_batch.reshape(-1, seq_len, feature_dim)
        query_x_flat = query_x_batch.reshape(-1, seq_len, feature_dim)

        #    调用父类的 embed 方法（实际上是调用 self.encoder.forward）
        #    support_z_flat: [B*NK, embed_dim]
        #    query_z_flat: [B*NQ, embed_dim]
        support_z_flat = self.embed(support_x_flat)
        query_z_flat = self.embed(query_x_flat)

        embed_dim = support_z_flat.size(-1)  # 获取编码后的嵌入维度

        #    将编码后的特征重新组织成按任务的结构
        #    support_z: [B, NK, embed_dim]
        #    query_z: [B, NQ, embed_dim]
        support_z_batch_view = support_z_flat.view(batch_size, num_support_per_task, embed_dim)
        query_z_batch_view = query_z_flat.view(batch_size, num_query_per_task, embed_dim)

        # support_y_batch 已经是 [B, NK]，可以直接在循环中使用 support_y_batch[i]

        all_task_logits_list = []  # 用于收集批次中每个任务计算得到的 logits

        # 2. 对批次中的每个任务独立进行原型计算和查询集分类
        for i in range(batch_size):  # 遍历元批次中的每一个任务
            # 提取当前任务 i 的支持集嵌入、支持集标签和查询集嵌入
            task_support_embeddings = support_z_batch_view[i]  # 形状: [NK, embed_dim]
            task_support_labels = support_y_batch[i]  # 形状: [NK]
            task_query_embeddings = query_z_batch_view[i]  # 形状: [NQ, embed_dim]

            prototypes_for_current_task = []  # 存储当前任务中计算得到的每个类别的原型

            # 为当前任务的 N 个类别分别计算原型
            for cls_idx in range(n_way):
                # 找到属于当前类别 cls_idx 的所有支持集样本的嵌入
                # task_support_labels == cls_idx 会产生一个布尔掩码
                class_mask = (task_support_labels == cls_idx)
                # 使用掩码从 task_support_embeddings 中选出对应类别的嵌入
                embeddings_for_this_class = task_support_embeddings[class_mask]  # 形状: [K, embed_dim]

                # 计算原型：该类别所有支持样本嵌入的均值
                if embeddings_for_this_class.nelement() == 0:
                    # 这是一个健壮性检查：理论上，如果 k_shot > 0 且数据采样正确，
                    # 每个类别都应该有 k_shot 个支持样本。
                    # 如果某个类没有支持样本，其原型无法计算。
                    # 可以用零向量作为回退，但这可能影响性能，并指示数据采样或任务构建逻辑存在问题。
                    print(f"警告: 任务 {i} (批内索引), 类别 {cls_idx} 没有找到支持样本。将使用零向量作为原型。")
                    prototype = torch.zeros(embed_dim, device=task_support_embeddings.device)
                else:
                    prototype = embeddings_for_this_class.mean(dim=0)  # 形状: [embed_dim]
                prototypes_for_current_task.append(prototype)

            # 将原型列表堆叠成一个张量
            # task_prototypes 形状: [N_way, embed_dim]
            task_prototypes = torch.stack(prototypes_for_current_task)

            # 3. 计算查询集样本到每个类原型的距离 (或相似度)
            #    我们使用欧氏距离的负值作为 logits。距离越小，负距离越大，表示越可能属于该类。
            #    task_query_embeddings: [NQ, embed_dim]
            #    task_prototypes: [N_way, embed_dim]
            #    torch.cdist(A, B) 会计算 A 中的每一行与 B 中的每一行之间的成对距离。
            #    所以，distances 的形状将是 [NQ, N_way]。
            distances = torch.cdist(task_query_embeddings, task_prototypes)

            # Logits: 距离的负值。交叉熵损失函数期望越高的logit代表越高的概率。
            task_logits = -distances  # 形状: [NQ, N_way]
            all_task_logits_list.append(task_logits)

        # 4. 将批次中所有任务的 logits 连接 (concatenate) 起来
        #    因为损失函数通常期望一个二维的输入 [总查询样本数, 类别数]
        #    总查询样本数 = batch_size * num_query_per_task
        #    所以最终 batch_logits 形状: [batch_size * NQ, N_way]
        batch_logits = torch.cat(all_task_logits_list, dim=0)

        return batch_logits