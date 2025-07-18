import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
import json
import os
from tqdm import tqdm
from solution import SVDNet, compute_loss, compute_ae_metric
from data_utils import ChannelDataPreprocessor, create_data_splits


def read_cfg_file(file_path):
    """读取配置文件"""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    samp_num = int(lines[0].strip())
    M = int(lines[1].strip())
    N = int(lines[2].strip())
    IQ = int(lines[3].strip())
    R = int(lines[4].strip())

    return samp_num, M, N, IQ, R


class HyperparameterTuner:
    """超参数调优器"""

    def __init__(self, scene_idx=1, round_idx=1):
        self.scene_idx = scene_idx
        self.round_idx = round_idx
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载数据
        self._load_data()

        # 搜索空间定义
        self.search_space = {
            'learning_rate': [1e-3, 2e-3, 5e-3, 1e-2],
            'batch_size': [2, 4, 8],
            'lambda_ortho': [0.1, 0.3, 0.5, 1.0],
            'lambda_singular': [0.01, 0.05, 0.1, 0.2],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'dropout_rate': [0.0, 0.1, 0.2],
        }

        self.results = []

    def _load_data(self):
        """加载数据"""
        data_dir = f"./CompetitionData{self.round_idx}"
        cfg_path = f"{data_dir}/Round{self.round_idx}CfgData{self.scene_idx}.txt"
        train_data_path = f"{data_dir}/Round{self.round_idx}TrainData{self.scene_idx}.npy"
        train_label_path = f"{data_dir}/Round{self.round_idx}TrainLabel{self.scene_idx}.npy"

        # 读取配置
        self.samp_num, self.M, self.N, self.IQ, self.R = read_cfg_file(cfg_path)

        # 加载数据
        train_data = np.load(train_data_path)
        train_labels = np.load(train_label_path)

        # 数据预处理
        preprocessor = ChannelDataPreprocessor()
        preprocessor.compute_statistics(train_data, train_labels)

        train_data = preprocessor.normalize_data(train_data)
        train_labels = preprocessor.normalize_labels(train_labels)

        # 数据分割
        splits = create_data_splits(train_data, train_labels, val_ratio=0.15, random_seed=42)

        self.train_data = splits['train_data']
        self.train_labels = splits['train_labels']
        self.val_data = splits['val_data']
        self.val_labels = splits['val_labels']

        print(f"Data loaded: Train={len(self.train_data)}, Val={len(self.val_data)}")

    def create_model_with_config(self, config):
        """根据配置创建模型"""
        # 这里可以根据配置修改模型结构
        model = SVDNet(M=self.M, N=self.N, R=self.R)

        # 应用dropout配置
        if hasattr(model, 'shared_features'):
            for module in model.shared_features:
                if isinstance(module, nn.Dropout):
                    module.p = config['dropout_rate']

        return model.to(self.device)

    def train_with_config(self, config, max_epochs=30):
        """使用给定配置训练模型"""
        # 创建模型
        model = self.create_model_with_config(config)

        # 优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=config['learning_rate'] / 10
        )

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 8

        for epoch in range(max_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_samples = 0

            # 随机打乱数据
            indices = np.random.permutation(len(self.train_data))

            for start_idx in range(0, len(indices), config['batch_size']):
                end_idx = min(start_idx + config['batch_size'], len(indices))
                batch_indices = indices[start_idx:end_idx]

                batch_loss = 0.0
                optimizer.zero_grad()

                for idx in batch_indices:
                    H_data = torch.FloatTensor(self.train_data[idx]).to(self.device)
                    H_label = torch.FloatTensor(self.train_labels[idx]).to(self.device)

                    U_out, S_out, V_out = model(H_data)

                    loss, _, _, _, _ = compute_loss(
                        U_out, S_out, V_out, H_label,
                        lambda_ortho=config['lambda_ortho'],
                        lambda_singular=config['lambda_singular']
                    )

                    batch_loss += loss

                batch_loss = batch_loss / len(batch_indices)
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += batch_loss.item() * len(batch_indices)
                train_samples += len(batch_indices)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_ae = 0.0
            val_samples = 0

            with torch.no_grad():
                for idx in range(len(self.val_data)):
                    H_data = torch.FloatTensor(self.val_data[idx]).to(self.device)
                    H_label = torch.FloatTensor(self.val_labels[idx]).to(self.device)

                    U_out, S_out, V_out = model(H_data)

                    loss, _, _, _, _ = compute_loss(
                        U_out, S_out, V_out, H_label,
                        lambda_ortho=config['lambda_ortho'],
                        lambda_singular=config['lambda_singular']
                    )

                    ae = compute_ae_metric(U_out, S_out, V_out, H_label)

                    val_loss += loss.item()
                    val_ae += ae
                    val_samples += 1

            avg_train_loss = train_loss / train_samples
            avg_val_loss = val_loss / val_samples
            avg_val_ae = val_ae / val_samples

            scheduler.step()

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return {
            'final_val_loss': avg_val_loss,
            'final_val_ae': avg_val_ae,
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }

    def grid_search(self, max_combinations=50):
        """网格搜索"""
        print("Starting hyperparameter grid search...")

        # 生成所有组合
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        all_combinations = list(itertools.product(*values))

        # 如果组合太多，随机采样
        if len(all_combinations) > max_combinations:
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            selected_combinations = [all_combinations[i] for i in indices]
        else:
            selected_combinations = all_combinations

        print(f"Testing {len(selected_combinations)} combinations...")

        pbar = tqdm(selected_combinations, desc="Grid Search")

        for combination in pbar:
            # 创建配置字典
            config = dict(zip(keys, combination))

            try:
                # 训练模型
                result = self.train_with_config(config)

                # 保存结果
                result['config'] = config
                self.results.append(result)

                # 更新进度条
                pbar.set_postfix({
                    'Best Val Loss': f"{result['best_val_loss']:.4f}",
                    'Val AE': f"{result['final_val_ae']:.4f}"
                })

            except Exception as e:
                print(f"Error with config {config}: {e}")
                continue

        # 按验证损失排序
        self.results.sort(key=lambda x: x['best_val_loss'])

        return self.results

    def random_search(self, num_trials=30):
        """随机搜索"""
        print("Starting hyperparameter random search...")

        pbar = tqdm(range(num_trials), desc="Random Search")

        for trial in pbar:
            # 随机采样配置
            config = {}
            for key, values in self.search_space.items():
                config[key] = np.random.choice(values)

            try:
                # 训练模型
                result = self.train_with_config(config)

                # 保存结果
                result['config'] = config
                result['trial'] = trial
                self.results.append(result)

                # 更新进度条
                pbar.set_postfix({
                    'Best Val Loss': f"{result['best_val_loss']:.4f}",
                    'Val AE': f"{result['final_val_ae']:.4f}"
                })

            except Exception as e:
                print(f"Error with config {config}: {e}")
                continue

        # 按验证损失排序
        self.results.sort(key=lambda x: x['best_val_loss'])

        return self.results

    def save_results(self, filename=None):
        """保存结果"""
        if filename is None:
            filename = f"hyperparameter_results_round{self.round_idx}_scene{self.scene_idx}.json"

        # 转换为可序列化的格式
        serializable_results = []
        for result in self.results:
            serializable_result = {
                'config': result['config'],
                'final_val_loss': float(result['final_val_loss']),
                'final_val_ae': float(result['final_val_ae']),
                'best_val_loss': float(result['best_val_loss']),
                'epochs_trained': int(result['epochs_trained'])
            }
            if 'trial' in result:
                serializable_result['trial'] = int(result['trial'])

            serializable_results.append(serializable_result)

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {filename}")

    def print_best_results(self, top_k=5):
        """打印最佳结果"""
        print(f"\nTop {top_k} hyperparameter configurations:")
        print("=" * 80)

        for i, result in enumerate(self.results[:top_k]):
            print(f"\nRank {i + 1}:")
            print(f"  Validation Loss: {result['best_val_loss']:.4f}")
            print(f"  Validation AE: {result['final_val_ae']:.4f}")
            print(f"  Epochs Trained: {result['epochs_trained']}")
            print("  Configuration:")
            for key, value in result['config'].items():
                print(f"    {key}: {value}")


def tune_hyperparameters(scene_idx=1, round_idx=1, method='random', num_trials=30):
    """超参数调优主函数"""
    print(f"Hyperparameter tuning for Scene {scene_idx}, Round {round_idx}")
    print(f"Method: {method}")

    tuner = HyperparameterTuner(scene_idx, round_idx)

    if method == 'grid':
        results = tuner.grid_search(max_combinations=num_trials)
    elif method == 'random':
        results = tuner.random_search(num_trials)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 保存和显示结果
    tuner.save_results()
    tuner.print_best_results()

    # 返回最佳配置
    if results:
        best_config = results[0]['config']
        print(f"\nBest configuration found:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")

        return best_config
    else:
        print("No valid results found!")
        return None


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for SVD Neural Network')
    parser.add_argument('--scene', type=int, default=1, help='Scene number')
    parser.add_argument('--round', type=int, default=1, help='Round number')
    parser.add_argument('--method', type=str, default='random', choices=['grid', 'random'], help='Search method')
    parser.add_argument('--trials', type=int, default=20, help='Number of trials')

    args = parser.parse_args()

    best_config = tune_hyperparameters(
        scene_idx=args.scene,
        round_idx=args.round,
        method=args.method,
        num_trials=args.trials
    )

    if best_config:
        print("\nYou can now use this configuration in your training script!")


if __name__ == "__main__":
    main()