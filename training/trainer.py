import torch
import torch.optim as optim  # 包含各种优化器，如 Adam, SGD
import torch.nn.functional as F  # 包含常用的函数，如损失函数 (cross_entropy), 激活函数等
from torch.utils.tensorboard import SummaryWriter  # 用于 TensorBoard 日志记录
from tqdm import tqdm  # 用于在循环中显示美观的进度条
import numpy as np
import os
import time  # 用于生成基于时间的唯一运行名称，避免日志和模型文件名冲突
import yaml  # 用于在TensorBoard中记录配置（如果config是字典或可以转换为字典）

# 从 utils 模块导入自定义的评估指标计算函数
from utils.metrics import calculate_accuracy_and_ci


# 辅助函数，用于从配置对象(可能是命名空间)创建字典，方便PyYAML序列化
def _config_to_dict(config_obj):
    if isinstance(config_obj, dict):
        return config_obj

    config_d = {}
    # 遍历配置对象的所有属性
    # vars(obj) 返回对象的 __dict__ 属性，即其属性和值的字典
    # hasattr(obj, '__dict__') 检查对象是否有 __dict__ 属性 (例如，简单类型如int, str没有)
    if hasattr(config_obj, '__dict__'):
        for k, v in vars(config_obj).items():
            if hasattr(v, '__dict__'):  # 如果属性值本身也是一个配置对象 (嵌套)
                config_d[k] = _config_to_dict(v)  # 递归转换
            elif isinstance(v, (list, tuple)):  # 处理列表/元组，确保内部元素也被转换
                config_d[k] = [_config_to_dict(item) if hasattr(item, '__dict__') else item for item in v]
            else:  # 基本类型
                config_d[k] = v
    else:  # 如果对象本身不是可转换的 (例如它就是一个基本类型值)，则直接返回
        return config_obj
    return config_d


class Trainer:
    """
    训练器 (Trainer) 类，封装了模型训练、验证、评估和日志记录的核心逻辑。
    这个类的设计目标是使其能够适应不同的小样本学习方法和配置。
    """

    def __init__(self,
                 model: torch.nn.Module,  # 要训练的小样本学习模型实例
                 train_loader: torch.utils.data.DataLoader,  # 训练数据加载器
                 val_loader: torch.utils.data.DataLoader,  # 验证数据加载器
                 test_loader: torch.utils.data.DataLoader,  # 测试数据加载器 (用于最终评估)
                 config: object  # 包含所有实验配置参数的对象
                 ):
        """
        初始化训练器。

        Args:
            model (torch.nn.Module): 实现了 FSL 逻辑的模型 (例如 PrototypicalNetwork 实例)。
            train_loader (DataLoader): 提供训练批次 (每批包含多个小样本任务)。
            val_loader (DataLoader): 提供验证批次。
            test_loader (DataLoader): 提供测试批次。
            config (object): 包含所有配置参数的对象 (通常是通过 YAML 文件加载的)。
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        # 确定运行设备 (CPU 或 GPU)
        self.device = self._get_device()
        self.model.to(self.device)  # 将模型参数和缓冲区移动到目标设备
        print(f"模型已移至设备: {self.device}")

        # 根据配置创建优化器和学习率调度器
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()

        # 初始化 TensorBoard SummaryWriter
        # 创建一个基于实验名称和当前时间戳的唯一日志目录
        # time.strftime('%Y%m%d-%H%M%S') 生成例如 "20230101-123000" 格式的时间戳
        current_time_str = time.strftime('%Y%m%d-%H%M%S')
        run_name_with_time = f"{config.experiment_name}_{current_time_str}"
        self.log_dir = os.path.join('runs', run_name_with_time)  # 日志通常保存在 'runs/' 目录下
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard 日志将保存至: {self.log_dir}")

        # 将实验配置记录到 TensorBoard 的 Text 标签页或 HParams (如果适用)
        try:
            config_as_dict = _config_to_dict(self.config)  # 将配置对象转换为字典
            # 使用 preformatted code block (Markdown的```yaml ... ```) 使其在TensorBoard中更易读
            config_text_for_tb = "```yaml\n" + yaml.dump(config_as_dict, indent=2, allow_unicode=True) + "\n```"
            self.writer.add_text("Experiment Configuration", config_text_for_tb, 0)
            print("实验配置已记录到 TensorBoard。")
        except Exception as e:
            print(f"警告: 记录配置到 TensorBoard 失败: {e}。可能是 PyYAML 未安装或配置对象结构复杂。")

        # 早停法 (Early Stopping) 相关变量初始化
        self.early_stopping_patience = config.training.early_stopping.patience  # 容忍验证损失不改善的轮数
        self.min_delta = config.training.early_stopping.min_delta  # 视为改善的最小变化量
        self.best_val_loss = float('inf')  # 初始化历史最佳验证损失为正无穷大
        self.epochs_no_improve = 0  # 记录验证损失连续未改善的轮数
        self.best_model_state = None  # 用于存储验证损失最低时的模型状态字典 (state_dict)

        self.global_train_step = 0  # 全局训练步数计数器，用于 TensorBoard 按步记录和学习率预热

    def _get_device(self) -> torch.device:
        """根据配置确定并返回 PyTorch 设备对象。"""
        if self.config.device.lower() == "auto":  # 配置为自动检测
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"自动检测设备: {'CUDA GPU 可用' if device_str == 'cuda' else '仅 CPU 可用'}")
            return torch.device(device_str)
        # 直接使用配置中指定的设备
        return torch.device(self.config.device)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """根据配置创建并返回优化器实例。"""
        opt_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate

        if opt_name == "adam":
            print(f"创建 Adam 优化器，学习率: {lr}")
            return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == "adamw":  # 如果未来支持 AdamW
            weight_decay = self.config.training.get('weight_decay', 0.01)  # 从配置获取，若无则默认0.01
            print(f"创建 AdamW 优化器，学习率: {lr}, 权重衰减: {weight_decay}")
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # 在此可以添加对其他优化器 (如 SGD) 的支持
        else:
            raise ValueError(f"不支持的优化器类型: '{self.config.training.optimizer}'。请在配置中选择 'Adam' 等。")

    def _create_lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """根据配置创建并返回学习率调度器实例。"""
        # 当前固定使用 ReduceLROnPlateau 调度器
        # 它会在验证集上的指标 (这里是 val_loss) 不再改善时，按因子降低学习率。
        scheduler_config = self.config.training.lr_scheduler
        print(
            f"创建 ReduceLROnPlateau 学习率调度器: factor={scheduler_config.factor}, patience={scheduler_config.patience}")
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # 监控的指标 (val_loss) 越小越好
            factor=scheduler_config.factor,  # 学习率降低的乘法因子
            patience=scheduler_config.patience,  # 容忍多少个 epoch 指标不改善后再降低学习率
            verbose=True  # 当学习率调整时打印信息
        )

    def _train_one_epoch(self, epoch_num: int) -> tuple[float, float]:
        """
        执行一个完整的训练轮次 (epoch)，包括前向传播、损失计算、反向传播和参数更新。

        Args:
            epoch_num (int): 当前的训练轮数 (从0开始计数)。

        Returns:
            tuple[float, float]: 该轮次的平均训练损失和平均训练准确率。
        """
        self.model.train()  # 1. 设置模型为训练模式
        # 这会启用 Dropout、BatchNorm 的训练行为（如更新移动均值/方差）。

        epoch_total_loss = 0.0  # 用于累加当前 epoch 中所有批次的损失
        epoch_total_acc = 0.0  # 用于累加当前 epoch 中所有批次的准确率

        # 使用 tqdm 创建一个进度条，迭代训练数据加载器
        # train_loader 每次迭代返回一个批次 (batch) 的小样本任务数据
        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch_num + 1}/{self.config.training.epochs} [训练中]",
                    unit="批次任务",  # 每个元素是一个批次，包含多个任务
                    leave=False)  # 迭代结束后不保留进度条的最终状态，避免多行输出

        for batch_idx, batch_of_tasks_data in enumerate(pbar):
            self.global_train_step += 1  # 更新全局训练步数

            # 2. 将数据移动到指定设备 (CPU/GPU)
            # batch_of_tasks_data 是一个元组: (support_x, support_y, query_x, query_y)
            # 每个张量的第一个维度是 meta-batch_size (即配置中的 training.batch_size)
            support_x, support_y, query_x, query_y = [tensor.to(self.device) for tensor in batch_of_tasks_data]

            # 3. 学习率预热 (Warmup) 逻辑
            # 在训练的初始阶段 (前 `warmup_steps` 步)，学习率从接近0线性增加到配置的目标学习率
            target_initial_lr = self.config.training.learning_rate
            current_optimizer_lr = target_initial_lr  # 默认使用目标学习率
            if self.global_train_step < self.config.training.warmup_steps:
                # 计算线性增加的学习率比例因子
                lr_scale_factor = float(self.global_train_step) / float(self.config.training.warmup_steps)
                current_optimizer_lr = target_initial_lr * lr_scale_factor
                # 将计算得到的预热学习率应用到优化器的所有参数组
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_optimizer_lr

            # 4. 清除优化器的梯度 (非常重要，否则梯度会累积)
            self.optimizer.zero_grad()

            # 5. 模型前向传播
            #    调用 FSL 方法 (如 PrototypicalNetwork) 的 forward 方法，处理这一个批次的任务。
            #    需要传入 n_way, k_shot, n_query 以便模型内部正确解析和处理数据。
            logits = self.model(support_x, support_y, query_x,
                                n_way=self.config.fsl_task.n_way,
                                k_shot=self.config.fsl_task.k_shot,
                                n_query=self.config.fsl_task.n_query)
            # `logits` 的期望形状: [meta_batch_size * n_way * n_query, n_way]

            # 6. 准备查询集标签以计算损失
            #    `query_y` 原始形状: [meta_batch_size, n_way * n_query]
            #    需要将其展平为一维张量以匹配 `logits` 的第一个维度和交叉熵损失的要求。
            query_y_flat = query_y.reshape(-1)  # 形状变为: [meta_batch_size * n_way * n_query]

            # 7. 计算损失
            #    使用交叉熵损失函数，适用于多分类问题。
            loss = F.cross_entropy(logits, query_y_flat)

            # 8. 反向传播 (计算梯度)
            loss.backward()

            # 9. 梯度裁剪 (Gradient Clipping)
            #    如果启用了梯度裁剪 (grad_clip_norm > 0)，则对模型参数的梯度进行裁剪，
            #    将其 L2 范数限制在 `grad_clip_norm` 以内，防止梯度爆炸。
            if self.config.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.config.training.grad_clip_norm)

            # 10. 参数更新 (执行一步优化)
            self.optimizer.step()

            # 11. 计算当前批次的准确率 (用于监控和日志)
            _, predicted_classes = torch.max(logits, 1)  # 获取每个查询样本预测的类别索引
            accuracy_batch = (predicted_classes == query_y_flat).float().mean().item()  # 计算准确率

            # 12. 累加损失和准确率
            epoch_total_loss += loss.item()
            epoch_total_acc += accuracy_batch

            # 13. TensorBoard 日志记录 (按训练步)
            self.writer.add_scalar('训练过程/损失 (每步)', loss.item(), self.global_train_step)
            self.writer.add_scalar('训练过程/准确率 (每步)', accuracy_batch, self.global_train_step)
            self.writer.add_scalar('学习率/当前值 (每步)', self.optimizer.param_groups[0]['lr'], self.global_train_step)

            # 更新进度条的后缀信息，显示当前批次的损失、准确率和学习率
            pbar.set_postfix({
                '损失': f"{loss.item():.4f}",
                '准确率': f"{accuracy_batch:.4f}",
                '学习率': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        pbar.close()  # 结束当前epoch的进度条

        # 计算整个 epoch 的平均损失和平均准确率
        avg_loss_epoch = epoch_total_loss / len(self.train_loader)
        avg_acc_epoch = epoch_total_acc / len(self.train_loader)

        # TensorBoard 日志记录 (按训练轮)
        self.writer.add_scalar('训练周期/平均损失 (每轮)', avg_loss_epoch, epoch_num)
        self.writer.add_scalar('训练周期/平均准确率 (每轮)', avg_acc_epoch, epoch_num)

        return avg_loss_epoch, avg_acc_epoch

    def _validate_one_epoch(self, epoch_num: int = None) -> tuple[float, float]:
        """
        在一个完整的验证集 (或测试集，如果调用时 `epoch_num` 为 `None`) 上评估模型性能。

        Args:
            epoch_num (int, optional): 当前的训练轮数。如果是最终测试评估，则可以为 `None`。
                                       主要用于区分日志记录的目标 (是常规验证还是最终测试)。

        Returns:
            tuple[float, float]: 该轮次的平均验证/测试损失和平均验证/测试准确率。
        """
        self.model.eval()  # 1. 设置模型为评估模式
        # 这会禁用 Dropout，并使 BatchNorm 层使用其在训练中累积的运行均值和方差。

        epoch_total_loss = 0.0
        epoch_total_acc = 0.0

        # 根据是验证阶段还是最终测试阶段，设置不同的进度条描述
        desc_prefix = f"Epoch {epoch_num + 1}/{self.config.training.epochs} [验证中]" if epoch_num is not None else "[最终评估中]"

        # 使用 val_loader (或 test_loader，如果从 evaluate_on_test_set 调用)
        # 这里为了通用性，假设传入的 loader 是 val_loader (在train()中调用时)
        # 或 test_loader (在evaluate_on_test_set()中调用时，但那里的逻辑更复杂)
        # 当前函数主要用于训练过程中的验证。
        current_loader = self.val_loader  # 在 train() 方法中调用此函数时，使用 val_loader

        pbar = tqdm(current_loader,
                    desc=desc_prefix,
                    unit="批次任务",
                    leave=False)

        # 2. 在 `torch.no_grad()` 上下文中执行评估，以禁用梯度计算，节省内存和计算资源
        with torch.no_grad():
            for batch_of_tasks_data in pbar:
                support_x, support_y, query_x, query_y = [tensor.to(self.device) for tensor in batch_of_tasks_data]

                # 模型前向传播
                logits = self.model(support_x, support_y, query_x,
                                    n_way=self.config.fsl_task.n_way,
                                    k_shot=self.config.fsl_task.k_shot,
                                    n_query=self.config.fsl_task.n_query)
                query_y_flat = query_y.reshape(-1)  # 展平查询集标签

                # 计算损失
                loss = F.cross_entropy(logits, query_y_flat)

                # 计算准确率
                _, predicted_classes = torch.max(logits, 1)
                accuracy_batch = (predicted_classes == query_y_flat).float().mean().item()

                epoch_total_loss += loss.item()
                epoch_total_acc += accuracy_batch
                pbar.set_postfix({'损失': f"{loss.item():.4f}", '准确率': f"{accuracy_batch:.4f}"})

        pbar.close()

        avg_loss_epoch = epoch_total_loss / len(current_loader)
        avg_acc_epoch = epoch_total_acc / len(current_loader)

        # 仅当是训练过程中的验证阶段 (epoch_num 不为 None) 时，才记录到 TensorBoard 的周期指标
        if epoch_num is not None:
            self.writer.add_scalar('验证周期/平均损失 (每轮)', avg_loss_epoch, epoch_num)
            self.writer.add_scalar('验证周期/平均准确率 (每轮)', avg_acc_epoch, epoch_num)

        return avg_loss_epoch, avg_acc_epoch

    def train(self):
        """
        执行完整的模型训练过程，包括所有配置的 epoch、学习率预热、学习率调度和早停。
        """
        print(f"--- 开始训练实验: {self.config.experiment_name} ---")
        print(f"目标设备: {self.device}")
        print(f"计划最大训练轮数: {self.config.training.epochs}")
        print(f"每轮训练批次数: {len(self.train_loader)}")
        print(f"每轮验证批次数: {len(self.val_loader)}")
        print("-" * 30)

        # 训练循环，最多执行 config.training.epochs 轮
        for epoch in range(self.config.training.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.training.epochs} 开始 ---")

            # 执行一轮训练
            avg_train_loss, avg_train_acc = self._train_one_epoch(epoch)

            # 执行一轮验证
            avg_val_loss, avg_val_acc = self._validate_one_epoch(epoch)

            # 获取当前 epoch 结束时的实际学习率 (可能已被预热或 ReduceLROnPlateau 调度器调整)
            actual_lr_this_epoch = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('学习率/每轮 (实际)', actual_lr_this_epoch, epoch)

            print(f"Epoch {epoch + 1}/{self.config.training.epochs} 完成 - "
                  f"训练损失: {avg_train_loss:.4f}, 训练准确率: {avg_train_acc:.4f} | "
                  f"验证损失: {avg_val_loss:.4f}, 验证准确率: {avg_val_acc:.4f} | "
                  f"当前学习率: {actual_lr_this_epoch:.7f}")

            # 学习率调度器根据当前验证损失调整学习率
            self.lr_scheduler.step(avg_val_loss)

            # --- 早停法 (Early Stopping) 逻辑 ---
            # 检查当前验证损失是否比历史最佳验证损失有显著改善
            if avg_val_loss < self.best_val_loss - self.min_delta:
                # 是的，找到了更好的模型状态
                self.best_val_loss = avg_val_loss  # 更新历史最佳验证损失
                self.epochs_no_improve = 0  # 重置连续未改善的轮数计数器
                # 深度拷贝当前模型的参数状态，作为新的最佳状态保存
                # model.state_dict() 返回一个包含模型所有可学习参数的字典
                self.best_model_state = self.model.state_dict().copy()
                print(f"*** 验证损失改善! 新的最佳验证损失: {self.best_val_loss:.4f}。已暂存模型状态。 ***")
            else:
                # 验证损失没有显著改善
                self.epochs_no_improve += 1
                print(
                    f"验证损失未显著改善 (已连续 {self.epochs_no_improve} / {self.early_stopping_patience} 轮未改善)。")
                # 检查是否达到了早停的耐心上限
                if self.epochs_no_improve >= self.early_stopping_patience:
                    print(f"--- 早停触发！已连续 {self.early_stopping_patience} 轮验证损失未显著改善。 ---")
                    break  # 跳出训练循环，提前结束训练
            # ---------------------------------

        print(f"\n--- 训练过程结束 ---")
        if self.best_model_state:
            print(f"加载在验证集上损失最低的模型状态 (验证损失为: {self.best_val_loss:.4f})。")
            self.model.load_state_dict(self.best_model_state)  # 将模型恢复到最佳状态
        else:
            # 如果从未找到比初始状态更好的模型 (例如，patience=0或val_loss一直上升)
            print("警告：在训练过程中未能记录到任何验证损失改善的状态。模型将保持训练结束时的最终状态。")

        # --- TensorBoard 记录超参数和与这些超参数对应的最终（最佳）指标 ---
        # 准备超参数字典，用于 add_hparams
        hparams_dict = {
            'learning_rate': self.config.training.learning_rate,
            'batch_size': self.config.training.batch_size,
            'encoder_type': self.config.encoder.type,
            'encoder_hidden_dim': self.config.encoder.hidden_dim,
            'encoder_num_layers': self.config.encoder.num_layers,
            'fsl_method': self.config.method.type,
            'fsl_n_way': self.config.fsl_task.n_way,
            'fsl_k_shot': self.config.fsl_task.k_shot,
            'fsl_n_query': self.config.fsl_task.n_query,
            'seq_len': self.config.data.seq_len,
            'warmup_steps': self.config.training.warmup_steps,
            'early_stop_patience': self.config.training.early_stopping.patience
        }
        self.hparams_dict = hparams_dict    # 将超参数字典hparams_dict存到实例属性以备后续调用
        # 准备与这些超参数对应的最终指标字典
        # 需要重新在验证集上评估已加载的最佳模型，以获取其准确率等指标
        # （因为 self.best_val_loss 只记录了损失，可能需要对应的准确率）
        # _validate_one_epoch(epoch_num=None) 表示这是一次非训练过程中的评估
        final_val_loss_of_best, final_val_acc_of_best = self._validate_one_epoch(epoch_num=None)

        metrics_for_hparams = {
            'hparam_metrics/best_validation_loss': self.best_val_loss,  # 实际上等于 final_val_loss_of_best
            'hparam_metrics/accuracy_at_best_val_loss': final_val_acc_of_best
        }
        try:
            # '.' 表示使用当前 TensorBoard run 的目录名作为 hparam 运行名
            self.writer.add_hparams(hparams_dict, metrics_for_hparams, run_name='.')
            print("超参数和最终验证指标已记录到 TensorBoard HParams。")
        except Exception as e:
            print(f"警告: TensorBoard HParams 日志记录失败: {e}。"
                  f"这通常发生在 PyTorch 版本较旧 (例如 < 1.3 或 1.6 之前对hparams支持不完善) "
                  f"或参数类型不完全匹配时。请检查或更新 PyTorch。")
        # ------------------------------------------------------------------

        self.writer.close()  # 完成所有写入后，关闭 SummaryWriter
        print("TensorBoard SummaryWriter 已关闭。")

    def evaluate_on_test_set(self):
        """
        在独立的测试集上评估训练好的模型 (通常是加载了最佳验证状态后的模型)，
        并报告 N-way K-shot 平均准确率及其 95% 置信区间。
        """
        num_target_test_episodes = self.config.evaluation.num_test_episodes
        print(f"\n--- 开始在测试集上进行最终评估 (目标任务数: {num_target_test_episodes}) ---")

        # 确保加载的是在验证集上表现最佳的模型状态
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print("已加载验证集上最佳模型状态进行测试评估。")
        else:
            print("警告: 未能在训练中找到更优模型状态。将使用训练结束时的模型进行测试评估。")

        self.model.eval()  # 设置模型为评估模式

        all_episode_accuracies = []  # 用于存储每个测试任务的准确率
        tasks_actually_processed = 0

        # 使用 tqdm 创建进度条，总数为目标评估的任务数
        pbar_test_episodes = tqdm(total=num_target_test_episodes, desc="[测试集评估]", unit="任务")

        # 从 test_loader 中迭代获取批次的任务数据
        # test_loader 的长度可能与 num_target_test_episodes 不直接对应
        # 我们需要从中采样 num_target_test_episodes 个任务
        for batch_of_tasks_data in self.test_loader:
            if tasks_actually_processed >= num_target_test_episodes:
                break  # 已处理足够数量的任务

            support_x, support_y, query_x, query_y = [tensor.to(self.device) for tensor in batch_of_tasks_data]

            with torch.no_grad():  # 在评估时禁用梯度计算
                # 模型前向传播，获取 logits
                logits_batch = self.model(support_x, support_y, query_x,
                                          n_way=self.config.fsl_task.n_way,
                                          k_shot=self.config.fsl_task.k_shot,
                                          n_query=self.config.fsl_task.n_query)
                # query_y 形状: [batch_size_tasks, Nway*Nquery]
                # logits_batch 形状: [batch_size_tasks * Nway*Nquery, Nway]
                query_y_flat_batch = query_y.reshape(-1)  # 展平查询集标签

                num_tasks_in_this_batch = support_x.size(0)  # 当前批次实际包含的任务数量
                num_queries_per_task = self.config.fsl_task.n_way * self.config.fsl_task.n_query

                # 分别计算这个批次中每个任务的准确率
                for i in range(num_tasks_in_this_batch):
                    if tasks_actually_processed >= num_target_test_episodes:
                        break  # 已达到目标任务数，即使当前批次还有剩余任务也不再处理

                    # 从批次 logits 和标签中提取当前任务 i 的部分
                    start_idx = i * num_queries_per_task
                    end_idx = (i + 1) * num_queries_per_task

                    task_logits = logits_batch[start_idx: end_idx]
                    task_query_y = query_y_flat_batch[start_idx: end_idx]

                    _, predicted_classes = torch.max(task_logits, 1)  # 获取预测类别
                    accuracy_episode = (predicted_classes == task_query_y).float().mean().item()  # 计算该任务的准确率
                    all_episode_accuracies.append(accuracy_episode)

                    tasks_actually_processed += 1
                    pbar_test_episodes.update(1)  # 更新进度条 (按已处理的任务数)

        pbar_test_episodes.close()

        if not all_episode_accuracies:  # 如果未能收集到任何准确率数据
            print("错误：在测试集评估中未能收集到任何任务的准确率。请检查测试 DataLoader 和评估逻辑。")
            # 可以在 TensorBoard 中记录测试失败状态
            self.writer.add_text("Test Evaluation Status", "Failed: No accuracies collected.", 0)
            return 0.0, 0.0

        # 使用 utils.metrics 中的函数计算平均准确率和 95% 置信区间
        mean_accuracy_test, ci_half_width_test = calculate_accuracy_and_ci(all_episode_accuracies)

        print(f"--- 测试集评估完成 ---")
        print(f"实际评估的任务数: {len(all_episode_accuracies)} / {num_target_test_episodes}")
        print(f"平均准确率: {mean_accuracy_test * 100:.2f}%")
        print(f"95% 置信区间半宽度: ±{ci_half_width_test * 100:.2f}%")
        lower_bound = (mean_accuracy_test - ci_half_width_test) * 100
        upper_bound = (mean_accuracy_test + ci_half_width_test) * 100
        print(f"准确率范围 (95% CI): ({lower_bound:.2f}%, {upper_bound:.2f}%)")

        # --- TensorBoard 日志记录 (测试集结果) ---
        self.writer.add_scalar('测试集评估/平均准确率', mean_accuracy_test, 0)  # global_step 设为0或其他标记值
        self.writer.add_scalar('测试集评估/95%CI半宽度', ci_half_width_test, 0)
        # 可以考虑再次记录 hparams 和测试集指标的组合，如果 TensorBoard HParams UI 支持多次记录
        # 或者创建一个新的 hparams 记录项专门用于测试结果
        test_metrics_for_hparams = {
            'hparam_final_test/accuracy': mean_accuracy_test,
            'hparam_final_test/ci_half_width': ci_half_width_test
        }
        # hparams_dict 可以是训练时用的同一份，或只包含关键识别参数
        try:
            # 使用不同的 metrics_for_hparams 来区分验证集上的最佳指标和测试集上的最终指标
            self.writer.add_hparams(self.hparams_dict, test_metrics_for_hparams, run_name='final_test_eval')
        except Exception as e:
            print(f"警告: TensorBoard HParams (测试集) 日志记录失败: {e}。")
        # --- --------------------------------- ---

        return mean_accuracy_test, ci_half_width_test