# 项目名称

FSL-Project-Test

---

## 目录

- [项目简介](#项目简介)
- [目录结构](#目录结构)
- [配置说明](#配置说明)
- [使用方式](#使用方式)
- [测试](#测试)

---

## 项目简介

本项目是对FSL-Traffic仓库的代码进行模块化尝试的测试仓库，主要是验证代码可行性，成品会同步到FSL-Traffic仓库中。



## 目录结构

```plaintext
<仓库根目录>/
├── configs/            # 配置文件（YAML）
├── data/               # 原始或预处理数据集
├── datasets/           # 自定义数据集定义
├── encoders/           # 编码器模型实现
├── methods/            # 小样本学习方法实现
├── runs/               # 训练及评估输出结果
├── training/           # 训练脚本及辅助工具
├── utils/              # 工具模块（指标计算、辅助函数等）
├── project_test/       # 实验性/原型代码
├── main.py             # 训练与评估入口脚本
├── requirements.txt    # Python 依赖列表
└── README.md           # 项目说明文档
```

请根据实际项目目录进行调整。


## 配置说明

编辑 `configs/proto_config.yaml`：

```yaml
experiment_name: my_experiment
# 设备，可选：auto、cpu、cuda
device: auto
training:
  epochs: 100      # 训练轮数
  batch_size: 16   # 每批任务数
  learning_rate: 0.001
  warmup_steps: 500
  early_stopping:
    patience: 10   # 验证损失不改善允许的轮数
    min_delta: 0.0001
  lr_scheduler:
    factor: 0.5    # 学习率衰减因子
    patience: 5    # 验证损失不改善允许的轮数
dataset:
  path: data/      # 数据集路径
fsl_task:
  n_way: 5         # N-way
  k_shot: 5        # K-shot
  n_query: 15      # 每类查询样本数
```

请根据项目需要自行修改。

## 使用方式

- **训练模型**：
  ```bash
  python main.py --config configs/proto_config.yaml
  ```

- **评估模型**：
  ```bash
  python main.py --config configs/proto_config.yaml --evaluate
  ```

## 测试

如果包含单元测试，可以这样运行：
```bash
pytest tests/
```
