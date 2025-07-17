# 🚀 SVD神经网络比赛项目启动指南

## 📁 项目目录结构

注：请确认克隆后svd_competition目录
```
svd_competition/
├── DebugData/
│   ├── Round0CfgData1.txt
│   ├── Round0CfgData2.txt
│   ├── Round0TestData1.npy
│   └── Round0TestData2.npy
├── CompetitionData1/
│   ├── Round1CfgData1.txt
│   ├── Round1CfgData2.txt  
│   ├── Round1CfgData3.txt
│   ├── Round1TrainData1.npy (640MB)
│   ├── Round1TrainData2.npy (640MB)
│   ├── Round1TrainData3.npy (640MB)
│   ├── Round1TrainLabel1.npy (640MB)
│   ├── Round1TrainLabel2.npy (640MB)
│   ├── Round1TrainLabel3.npy (640MB)
│   ├── Round1TestData1.npy (32MB)
│   ├── Round1TestData2.npy (32MB)
│   └── Round1TestData3.npy (32MB)
├── system_check.py         # 环境检查脚本
├── solution.py             # 比赛要求的模型文件（自己要修改）
├── demo_code.py            # 组委会提供的测试脚本（我们用test_and_submit.py代替，目前不使用）
├── simple_train.py         # 训练脚本
├── test_and_submit.py      # 测试和提交脚本
└── requirements.txt        # 依赖包列表
```
## 克隆仓库

```bash
# 在自选目录下运行以下代码
# 请选择全英文路径文件夹克隆仓库，符号尽量只含有下划线，如“D:\PyWorks\”
git clone https://github.com/YOUR_USERNAME/svd_competition.git
# 进入文件目录
cd svd_competition
```

## 环境准备

### 1. 创建虚拟环境

```bash
# 在svd_competition目录下运行以下代码
# 请按照需求自行配置，这里给出虚拟环境
# 创建虚拟环境
python -m venv svd_env

# 激活虚拟环境
# Windows:
svd_env\Scripts\activate
# Linux/Mac:
source svd_env/bin/activate
```

### 2. 安装依赖

```bash
# 请确认依赖文件配置，特别是CUDA配置，详情参见requirements.txt内部
pip install -r requirements.txt
```

### 3. 检查系统

```bash
#在svd_competition目录下运行以下代码，请确认使用GPU加速
python system_check.py
```


## 训练及测试

### 1. 模型训练

注： --scene代表场景 --round代表比赛轮次（1：初赛，2：复赛，目前使用初赛）

训练所有场景、依次训练三个场景 二选一

---

训练所有场景（使用此条即可不使用依次训练）

```bash
# 在终端svd_competition目录下运行以下代码
python simple_train.py --round 1
# 使用: Round1CfgData1.txt + Round1TrainData1.npy + Round1TrainLabel1.npy
# 生成: svd_model_round1_scene1.pth
# 使用: Round1CfgData2.txt + Round1TrainData2.npy + Round1TrainLabel2.npy
# 生成: svd_model_round1_scene2.pth
# 使用: Round1CfgData3.txt + Round1TrainData3.npy + Round1TrainLabel3.npy 
# 生成: svd_model_round1_scene3.pth
```

依次训练三个场景

```bash
# 在终端svd_competition目录下运行以下代码
python simple_train.py --scene 1 --round 1
python simple_train.py --scene 2 --round 1  
python simple_train.py --scene 3 --round 1
```

### 2. 模型测试

测试所有场景（使用此条即可不使用依次测试）

```bash
# 在svd_competition目录下运行以下代码
python test_and_submit.py --round 1
# 使用: 所有Round1CfgData*.txt + Round1TestData*.npy + 训练好的.pth文件
# 生成: Round1TestOutput1.npz, Round1TestOutput2.npz, Round1TestOutput3.npz
```

依次测试三个场景

```bash
# 在svd_competition目录下运行以下代码
python test_and_submit.py --scene 1 --round 1
python test_and_submit.py --scene 2 --round 1
python test_and_submit.py --scene 3 --round 1
```

注：生成的.pth和.npz文件直接在svd_competition目录下

## 结果提交

### 1. 创建提交包

```bash
python test_and_submit.py --round 1 --submit
```

zip文件确认包含：

```
submission_round1.zip
├── Round1TestOutput1.npz
├── Round1TestOutput2.npz  
├── Round1TestOutput3.npz
├── solution.py
├── svd_model_round1_scene1.pth
├── svd_model_round1_scene2.pth
└── svd_model_round1_scene3.pth
```

## 优化策略

### 1.模型架构优化、损失函数优化

修改solution.py

### 2.训练策略优化

修改simple_train.py

### 3.备注

80%的优化只需要改 `solution.py`

先从改进模型架构开始，这样最安全也最有效果！
