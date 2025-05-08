import torch
import numpy as np
import random


def set_seeds(seed: int):
    """
    设置全局随机种子以确保 PyTorch、NumPy 和 Python 内置 random 模块的实验结果可复现性。
    在机器学习实验中，许多操作（如权重初始化、数据打乱、dropout等）都依赖于随机数生成。
    通过固定种子，可以使得这些随机过程在每次运行时产生相同的结果，便于调试和比较。

    Args:
        seed (int): 要设置的整数种子值。
    """
    # 为 PyTorch 在 CPU 上的操作设置种子
    torch.manual_seed(seed)

    # 如果 CUDA (GPU) 可用，也为 GPU 操作设置种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前活动的 GPU 设置种子
        torch.cuda.manual_seed_all(seed)  # 如果你使用了多个 GPU，为所有 GPU 设置种子

    # 为 NumPy 设置种子
    np.random.seed(seed)

    # 为 Python 内置的 random 模块设置种子
    random.seed(seed)

    # --- 关于 CuDNN 的确定性设置 ---
    # CuDNN 是 NVIDIA 提供的用于深度神经网络的 GPU 加速库。
    # 它包含一些优化算法，这些算法在某些情况下可能会引入不确定性以换取性能。
    # 为了完全的可复现性，通常需要进行以下设置：

    # torch.backends.cudnn.benchmark = False
    # 当设置为 True 时，CuDNN 会在每次前向传播时运行基准测试，以找到针对当前输入尺寸的最优卷积算法。
    # 这可以提高性能，但由于算法选择可能变化，会导致结果不一致。设为 False 以禁用。

    # torch.backends.cudnn.deterministic = True
    # 当设置为 True 时，指示 CuDNN 仅使用确定性的卷积算法，并禁用那些非确定性的算法。
    # 这有助于确保可复现性，但可能会牺牲一些性能。

    # 注意: 即使进行了这些设置，完全的跨平台/跨版本可复现性有时仍然具有挑战性。
    # 但这些步骤是实现可复现性的重要基础。
    # 在你的 YAML 配置中已经设置了种子，这里是实际应用它的地方。
    # 为了更高的确定性，可以将上面两行取消注释，但你的实验目前在CPU上，这两行主要影响GPU。
    # 如果切换到GPU，可以考虑启用它们以追求极致的可复现性。