import numpy as np
import scipy.stats  # 导入 scipy.stats 以便使用 t 分布计算置信区间


def calculate_accuracy_and_ci(accuracies: list) -> tuple[float, float]:
    """
    根据一组准确率观测值（例如，来自多个小样本测试任务的准确率），
    计算它们的平均准确率和 95% 置信区间的半宽度。

    Args:
        accuracies (list): 一个包含浮点数准确率的列表，每个值都在 0.0 到 1.0 之间。
                           列表中的每个元素代表一次独立评估（例如一个测试 episode）的准确率。

    Returns:
        tuple[float, float]:
            - 第一个元素: 平均准确率 (mean_accuracy)。
            - 第二个元素: 95% 置信区间的半宽度 (confidence_interval_half_width)。
              最终的置信区间可以表示为 `mean_accuracy ± confidence_interval_half_width`。
    """
    # 检查输入列表是否为空
    if not accuracies:
        print("警告 (calculate_accuracy_and_ci): 准确率列表为空，无法计算统计数据。返回 (0.0, 0.0)。")
        return 0.0, 0.0

    accuracies_np = np.array(accuracies, dtype=np.float64)  # 转换为 NumPy 数组以便进行数值计算，使用float64提高精度

    # 计算平均准确率
    mean_acc = np.mean(accuracies_np)

    # 获取样本数量 (即评估的任务/episodes 数量)
    n_samples = len(accuracies_np)

    # 如果只有一个样本点，标准差和置信区间无明确定义或意义不大
    if n_samples <= 1:
        # print("警告 (calculate_accuracy_and_ci): 只有一个或没有准确率样本，置信区间半宽度设为 0.0。")
        return mean_acc, 0.0  # 返回均值，置信区间宽度为0

    # 计算标准差
    std_dev = np.std(accuracies_np, ddof=0)  # ddof=0 计算总体标准差，ddof=1 计算样本标准差。对于大量样本，差异不大。
    # 在FSL论文中，通常样本量（episodes）较大。

    # 计算均值的标准误差 (Standard Error of the Mean, SEM)
    # SEM = std_dev / sqrt(n_samples)
    std_err = std_dev / np.sqrt(n_samples)

    # 计算 95% 置信区间的半宽度
    # 置信区间表示真实平均准确率有 95% 的概率落在 [均值 - 半宽度, 均值 + 半宽度] 范围内。
    # 当样本量 n_samples 较大时 (例如 > 30 或更多)，可以使用正态分布的 Z 分数 (1.96 for 95% CI)。
    # 当样本量较小时，使用 t 分布的临界值更为准确。
    # t 分布的临界值取决于置信水平 (95%) 和自由度 (df = n_samples - 1)。

    # 为 alpha/2 (双尾检验中每条尾巴的概率)
    # 对于 95% 置信度，alpha = 0.05, 所以每条尾巴是 0.025。
    # 我们需要找到使得累积分布函数 (CDF) 值为 1 - 0.025 = 0.975 的 t 值。
    critical_value = scipy.stats.t.ppf(0.975, df=n_samples - 1)

    # 有时对于极小的n_samples，scipy.stats.t.ppf 可能返回nan或inf。
    # 或者在FSL论文中，即使n_samples不是特别大（如600个episodes），也常直接用1.96作为近似。
    if np.isnan(critical_value) or np.isinf(critical_value) or n_samples >= 600:  # 600是个拍脑袋的较大值
        critical_value = 1.96  # 使用正态分布近似
        # print("信息 (calculate_accuracy_and_ci): 使用 1.96 (正态分布近似) 计算置信区间。")

    confidence_interval_half_width = critical_value * std_err

    return mean_acc, confidence_interval_half_width