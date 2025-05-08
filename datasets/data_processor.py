# 该文件负责数据的原始加载和基础预处理。
# 主要功能是从 Excel 文件中读取数据，进行合并、填充缺失值、特征选择和标准化。

import pandas as pd  # Pandas 用于高效地处理表格数据 (如 Excel, CSV)
from sklearn.preprocessing import StandardScaler  # Scikit-learn 用于标准的机器学习预处理步骤，如特征标准化


def load_and_preprocess_data(incident_path: str, before_path: str, after_path: str) -> tuple:
    """
    从指定的 Excel 文件路径加载事件数据、事件前数据和事件后数据，
    进行合并、标签化、特征选择、缺失值填充和特征标准化。

    Args:
        incident_path (str): 包含事件（通常是异常）样本的 Excel 文件路径。
        before_path (str): 包含事件发生前（通常是正常）样本的 Excel 文件路径。
        after_path (str): 包含事件发生后（通常是正常）样本的 Excel 文件路径。

    Returns:
        tuple: 包含以下元素的元组：
            - features_scaled (numpy.ndarray): 标准化后的特征数据，形状为 [总样本点数, 特征维度]。
                                             每一行是一个时间点，每一列是一个特征。
            - labels (numpy.ndarray): 对应每个时间点的标签 (0 表示正常, 1 表示异常)，形状为 [总样本点数]。
            - feature_cols (list[str]): 使用的特征列的名称列表。
            - scaler (sklearn.preprocessing.StandardScaler): 用于标准化的 `StandardScaler` 对象实例。
                                                          可以保存下来用于未来对新数据进行相同的标准化转换，
                                                          或者用于将标准化后的数据逆转换为原始尺度。
    """
    print(f"开始加载数据: incident='{incident_path}', before='{before_path}', after='{after_path}'")
    # 1. 使用 Pandas 加载 Excel 数据到 DataFrame 对象
    try:
        incident_df = pd.read_excel(incident_path)
        before_df = pd.read_excel(before_path)
        after_df = pd.read_excel(after_path)
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到。请检查路径。 {e}")
        raise
    print("数据加载完成。")

    # 2. 特征列提取
    # 假设以 'Data' 开头的列是主要的数值型特征
    data_cols = [col for col in incident_df.columns if col.startswith('Data')]
    # 假设 'Time_tag', 'week', 'Percentage', 'Median' 是其他需要考虑的时间相关特征
    time_features = ['Time_tag', 'week', 'Percentage', 'Median']
    # 组合所有要使用的特征列名
    feature_cols = data_cols + time_features
    print(f"使用的特征列: {feature_cols}")

    # 3. 标签设置 (Label Assignment)
    # 为不同来源的数据帧分配标签：1 代表异常（事件），0 代表正常（事件前后）
    incident_df['label'] = 1  # 异常样本
    before_df['label'] = 0  # 正常样本
    after_df['label'] = 0  # 正常样本
    print("标签分配完成。")

    # 4. 合并数据集 (Concatenate Datasets)
    # 将三个数据帧按行合并成一个大的数据帧
    # ignore_index=True 会重新生成从0开始的索引
    data_df = pd.concat([incident_df, before_df, after_df], ignore_index=True)
    print(f"数据合并完成。总行数: {len(data_df)}")

    # 5. 缺失值处理 (Handling Missing Values)
    # 使用前一个有效值 (forward fill) 来填充 NaN (Not a Number) 值
    # 这是一种简单的时间序列数据缺失值填充方法，假设数据在短时间内变化不大
    # 如果缺失值情况复杂，可能需要更高级的填充策略 (如插值、基于模型的填充等)
    original_nan_count = data_df[feature_cols].isnull().sum().sum()
    data_df = data_df.ffill()
    filled_nan_count = data_df[feature_cols].isnull().sum().sum()
    if original_nan_count > 0:
        print(
            f"缺失值处理: 使用 ffill 填充了 {original_nan_count - filled_nan_count} 个 NaN 值。剩余 NaN: {filled_nan_count}")
    if filled_nan_count > 0:
        print(f"警告: 特征列中仍有 {filled_nan_count} 个 NaN 值。可能需要进一步处理或检查数据源。")
        # 对于仍存在的NaN（例如文件开头的NaN无法ffill），可以考虑用0填充或均值填充
        data_df[feature_cols] = data_df[feature_cols].fillna(0)  # 示例：用0填充剩余NaN
        print(f"剩余的 {filled_nan_count} 个 NaN 值已用 0 填充。")

    # 6. 特征标准化 (Feature Scaling)
    # StandardScaler 会将每个特征列的数据转换为均值为0，标准差为1的分布。
    # 这对于许多机器学习算法（包括神经网络）的稳定训练和良好性能至关重要。
    scaler = StandardScaler()

    # 从 data_df 中提取所有定义好的特征列
    features_to_scale = data_df[feature_cols]

    # 使用 .fit_transform() 方法：
    # .fit() 会计算特征列的均值和标准差。
    # .transform() 会使用计算得到的均值和标准差来标准化数据。
    features_scaled = scaler.fit_transform(features_to_scale)

    # 提取标签列为 NumPy 数组
    labels = data_df['label'].values
    print("特征标准化完成。")
    print(f"最终特征数据形状: {features_scaled.shape}, 标签数据形状: {labels.shape}")

    # 返回处理后的数据和相关信息
    return features_scaled, labels, feature_cols, scaler