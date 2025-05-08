# utils 模块的初始化文件。
# 这个文件可以保持为空，或者用于方便地从 utils 包中导入其子模块的常用内容。
# 例如，如果 metrics.py 和 torch_utils.py 中有你想直接通过 from utils import ... 访问的函数，
# 可以在这里进行导入:
#
# from .metrics import calculate_accuracy_and_ci
# from .torch_utils import set_seeds
#
# 这样做可以简化外部模块调用 utils 中函数时的路径。
# 目前，我们让调用方直接指定子模块，如 from utils.metrics import ...