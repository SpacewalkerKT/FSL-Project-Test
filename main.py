import yaml  # 用于加载和解析 YAML 格式的配置文件
import argparse  # 用于从命令行读取参数，例如配置文件的路径
import os  # 用于操作系统相关功能，如检查文件是否存在，创建目录
import torch  # PyTorch 深度学习框架
import time  # 用于生成基于时间戳的唯一文件名或目录名

# --- 从自定义模块导入必要的类和函数 ---
# 工具类
from utils.torch_utils import set_seeds  # 用于设置随机种子，确保实验可复现
# 数据处理与加载
from datasets.dataloaders import get_dataloaders  # 获取训练、验证、测试数据加载器的函数
# 编码器模型
from encoders.time_series_encoder import TimeSeriesEncoder  # 我们当前使用的时间序列 Transformer 编码器
# 如果未来添加其他编码器，例如:
# from encoders.cnn_encoder import CNNEncoder
# 小样本学习方法模型
from methods.prototypical_network import PrototypicalNetwork  # 我们当前使用的原型网络方法
# 如果未来添加其他方法，例如:
# from methods.maml import MAML
# 训练器
from training.trainer import Trainer  # 封装了训练和评估逻辑的训练器类


# --- ----------------------------- ---

# 定义一个辅助类 ConfigNamespace，用于将从 YAML 加载的字典转换为可以通过点操作符访问属性的对象。
# 例如，可以直接使用 `config.data.seq_len` 而不是 `config['data']['seq_len']`。
# 这使得在代码中访问配置参数更简洁易读。
class ConfigNamespace:
    def __init__(self, config_dict: dict):
        """
        通过字典递归地构造 ConfigNamespace 对象。
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):  # 如果值本身也是一个字典，则递归地创建 ConfigNamespace
                setattr(self, key, ConfigNamespace(value))
            else:  # 否则，直接设置属性
                setattr(self, key, value)

    def __repr__(self, indent_level: int = 0) -> str:
        """
        自定义对象的字符串表示形式，使其在打印时更易读，能显示层级结构。
        """
        indent = "  " * indent_level
        attrs = []
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNamespace):
                # 递归调用 __repr__ 并增加缩进
                attrs.append(f"{indent}{k}:\n{v.__repr__(indent_level + 1)}")
            else:
                attrs.append(f"{indent}{k}: {v}")
        return "\n".join(attrs)


def main(config_path: str):
    """
    实验的主执行函数。
    它负责：
    1. 加载配置。
    2. 设置随机种子。
    3. 准备数据加载器。
    4. 根据配置初始化编码器和小样本学习模型。
    5. 初始化并运行训练器进行模型训练和评估。
    6. (可选) 保存最终的最佳模型。

    Args:
        config_path (str): 指向 YAML 配置文件的路径。
    """
    # 1. 加载 YAML 配置文件
    print(f"步骤 1: 从 '{config_path}' 加载实验配置...")
    if not os.path.exists(config_path):
        print(f"错误: 配置文件 '{config_path}' 不存在。程序将退出。")
        return

    with open(config_path, 'r', encoding='utf-8') as f:  # 使用 utf-8 编码打开文件
        try:
            config_dict = yaml.safe_load(f)  # safe_load 比 load 更安全
        except yaml.YAMLError as exc:
            print(f"错误: 解析 YAML 配置文件 '{config_path}' 失败。")
            if hasattr(exc, 'problem_mark'):
                print(f"  错误位置: 行 {exc.problem_mark.line + 1}, 列 {exc.problem_mark.column + 1}")
            print(f"  具体错误: {exc}")
            return

    # 将加载的字典转换为 ConfigNamespace 对象，方便后续通过点操作符访问
    config = ConfigNamespace(config_dict)
    print("配置加载完成。配置内容如下:")
    print(str(config))  # 使用自定义的 __repr__ 打印配置
    print("-" * 30)

    # 2. 设置随机种子
    # 为了实验的可复现性，在所有随机操作开始前设置种子。
    print(f"步骤 2: 设置全局随机种子为 {config.random_seed}...")
    set_seeds(config.random_seed)
    print("随机种子设置完成。")
    print("-" * 30)

    # 3. 获取数据加载器 (DataLoaders)
    # `get_dataloaders` 函数会负责加载原始数据、预处理、创建 FewShotDataset 实例，
    # 并最终返回训练、验证和测试集的 DataLoader。
    # 重要：该函数还会从数据中推断出 `feature_dim` 并将其更新到 `config.encoder` 对象中。
    print("步骤 3: 准备数据加载器...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"数据加载器准备完成。编码器所需的 feature_dim 已从数据中确定为: {config.encoder.feature_dim}")
    print("-" * 30)

    # 4. 初始化编码器 (Encoder)
    # 根据配置文件中 `config.encoder.type` 的值来选择并实例化相应的编码器。
    print(f"步骤 4: 初始化编码器 (类型: {config.encoder.type})...")
    if config.encoder.type == "TimeSeriesTransformer":
        encoder = TimeSeriesEncoder(
            feature_dim=config.encoder.feature_dim,  # 此值已由 get_dataloaders 动态设定
            hidden_dim=config.encoder.hidden_dim,
            num_layers=config.encoder.num_layers,
            nhead=config.encoder.nhead,
            dropout=config.encoder.dropout
        )
    # elif config.encoder.type == "CNNEncoder": # 示例：如果未来添加了 CNNEncoder
    #     encoder = CNNEncoder(feature_dim=config.encoder.feature_dim, **vars(config.encoder_specific_params_for_cnn))
    else:
        raise ValueError(f"不支持的编码器类型: '{config.encoder.type}'。请检查配置文件。")
    print(f"编码器 '{config.encoder.type}' 初始化完成。")
    print("-" * 30)

    # 5. 初始化小样本学习方法 (FSL Method)
    # 根据配置文件中 `config.method.type` 的值选择并实例化 FSL 方法模型。
    # 编码器实例会作为参数传递给 FSL 方法的构造函数。
    print(f"步骤 5: 初始化小样本学习方法 (类型: {config.method.type})...")
    if config.method.type == "PrototypicalNetwork":
        model = PrototypicalNetwork(encoder=encoder)
    # elif config.method.type == "MAML": # 示例：如果未来添加了 MAML
    #     # MAML 可能需要更复杂的模型结构，例如编码器后接一个分类头，整个作为MAML模型
    #     # 这里假设 MAML 类能处理编码器和它自己的特定参数
    #     maml_specific_params = config.method_params.maml if hasattr(config.method_params, 'maml') else {}
    #     model = MAML(encoder=encoder, **vars(maml_specific_params)) # vars()用于从Namespace提取参数字典
    else:
        raise ValueError(f"不支持的小样本学习方法类型: '{config.method.type}'。请检查配置文件。")
    print(f"小样本学习方法 '{config.method.type}' 初始化完成。")
    print("-" * 30)

    # 6. 初始化训练器 (Trainer)
    # Trainer 类封装了整个训练和评估流程。
    print("步骤 6: 初始化训练器...")
    trainer = Trainer(
        model=model,  # 完整的 FSL 模型 (已包含编码器)
        train_loader=train_loader,  # 训练数据加载器
        val_loader=val_loader,  # 验证数据加载器
        test_loader=test_loader,  # 测试数据加载器 (用于最终评估)
        config=config  # 完整的配置对象
    )
    print("训练器初始化完成。")
    print("-" * 30)

    # 7. 开始训练模型
    # `trainer.train()` 方法将执行包括学习率预热、多轮训练、验证、学习率调度、早停等在内的完整流程。
    # 训练结束后，`trainer.model` 将被设置为在验证集上表现最佳的模型状态。
    print("步骤 7: 开始模型训练...")
    trainer.train()
    print("模型训练已结束。")
    print("-" * 30)

    # 8. 在测试集上进行最终评估
    # 使用训练好的模型（已加载最佳验证状态）在独立的测试集上进行评估，
    # 报告 N-way K-shot 平均准确率和 95% 置信区间。
    print("步骤 8: 在测试集上进行最终评估...")
    trainer.evaluate_on_test_set()
    print("测试集评估已结束。")
    print("-" * 30)

    # 9. (可选) 保存最终使用的模型状态 (即验证集上表现最佳的模型)
    # trainer.model 在 train() 结束后已经是最佳状态，这里是显式保存到文件。
    # TensorBoard 的日志目录名包含了时间戳，可以用来创建唯一的模型保存路径。
    # trainer.log_dir 是 TensorBoard 日志的完整路径
    model_dir = trainer.log_dir  # 直接使用 TensorBoard 的 run 目录作为模型保存目录

    # 确保模型保存目录存在
    # os.makedirs(model_dir, exist_ok=True) # trainer中已创建，这里可以省略

    model_save_filename = "best_model_on_validation.pth"
    model_save_path = os.path.join(model_dir, model_save_filename)

    if trainer.best_model_state:  # 检查早停法是否找到了一个“最佳”状态
        # 保存的是 state_dict，只包含模型参数，不包含完整模型结构
        # 加载时需要先创建模型实例，再 model.load_state_dict(...)
        torch.save(trainer.best_model_state, model_save_path)
        print(f"验证集上表现最佳的模型状态已保存至: {model_save_path}")
    else:
        # 如果早停从未触发，或者验证损失从未改善过初始状态 (理论上不太可能，除非patience极小)
        # 这种情况下，trainer.model 是训练到最后一轮的状态
        final_model_path = os.path.join(model_dir, "model_at_training_end.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"警告: 早停机制可能未找到优于初始状态的模型，或者训练完成了所有轮次。"
              f"训练结束时的模型状态已保存至: {final_model_path}")
    print("-" * 30)
    print("主程序执行完毕。")


if __name__ == "__main__":
    # 1. 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="模块化小样本学习实验框架")

    # 2. 添加 '--config' 参数，用于指定配置文件的路径
    #    default 值设定了如果没有从命令行提供此参数时的默认配置文件路径。
    parser.add_argument(
        '--config',
        type=str,
        default="configs/proto_config.yaml",  # 默认配置文件
        help="指向 YAML 配置文件的路径 (例如: configs/proto_config.yaml)"
    )
    # 3. 解析命令行参数
    args = parser.parse_args()

    # 4. 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 '{args.config}' 未找到。请确保文件路径正确，或已创建该配置文件。")
        print("程序将退出。")
    else:
        # 5. 如果配置文件存在，则调用 main 函数开始实验
        main(args.config)