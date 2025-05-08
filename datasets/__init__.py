# datasets 模块的初始化文件。
# 当其他 Python脚本通过 'import datasets' 或 'from datasets import ...' 的方式导入此目录时，
# 这个 __init__.py 文件会被首先执行。

# 目前，这个文件可以保持为空。
# 如果未来你希望在导入 datasets 模块时自动执行某些操作，或者提供更简洁的导入路径，
# 可以在这里添加代码。例如：
#
# from .data_processor import load_and_preprocess_data
# from .few_shot_dataset import FewShotDataset
# from .dataloaders import get_dataloaders
#
# 这样，其他文件就可以直接写:
# from datasets import load_and_preprocess_data
# 而不是:
# from datasets.data_processor import load_and_preprocess_data
#
# 这是一种常见的 Python 包组织方式，可以使导入更清晰。
# 对于我们目前的结构，保持为空也是完全可以的。