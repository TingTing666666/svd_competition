# SVD竞赛main_stable_multi_test启动指南

## 4个版本可以同时训练：

### 方法1: 单独训练特定版本
```bash
# 训练v1版本(强正交化)
python multi_train.py --version v1 --scene 1 --round 1

# 训练v2版本(注意力增强)
python multi_train.py --version v2 --scene 1 --round 1

# 训练v3版本(轻量高效)
python multi_train.py --version v3 --scene 1 --round 1

# 原始版本
python simple_train.py --scene 1 --round 1
```

### 方法2: 自动对比所有版本
```bash
# 对单个场景训练所有版本并自动选择最佳
python multi_train.py --scene 1 --round 1

# 对所有场景训练所有版本
python multi_train.py --all --round 1
```

### 方法3: 并行启动(多终端)
```bash
# 终端1
python multi_train.py --version v1 --scene 1 --round 1 &

# 终端2  
python multi_train.py --version v2 --scene 1 --round 1 &

# 终端3
python multi_train.py --version v3 --scene 1 --round 1 &

# 终端4
python simple_train.py --scene 1 --round 1 &
```

### 各版本特点

- **v1 (强正交化)**: 多次迭代正交化，严格约束，可能AE最低但训练慢
- **v2 (注意力增强)**: 注意力机制+残差连接，表达能力强
- **v3 (轻量高效)**: 参数少训练快，适合快速验证
- **原版**: 中等复杂度的baseline

系统会自动选择最佳版本并复制为标准命名，然后用test_and_submit.py测试即可。