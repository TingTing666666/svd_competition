# SVD竞赛启动指南

## 快速开始

### 1. 训练模型
```bash
# 训练单个场景
python simple_train.py --scene 1 --round 1

# 训练所有场景
python simple_train.py --round 1
```

### 2. 测试并生成提交文件
```bash
# 测试所有场景并创建提交包
python test_and_submit.py --round 1 --submit
```

## 文件说明

- **solution.py**: 核心SVD神经网络模型
- **simple_train.py**: 训练脚本
- **test_and_submit.py**: 测试和提交文件生成

## 训练流程

1. 将比赛数据放在 `./CompetitionData1/` 目录下
2. 运行训练命令，会自动生成 `svd_model_round1_scene*.pth` 模型文件
3. 运行测试命令，会生成 `Round1TestOutput*.npz` 结果文件
4. 最终生成 `submission_round1.zip` 提交包

## 注意事项

- 确保有CUDA环境以加速训练
- 单个场景训练约需30-50轮
- 模型专注提升AE分数，移除了所有无关代码