import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import math
from solution import SVDNet, compute_loss, compute_ae_metric


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


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class WarmupCosineScheduler:
    """带预热的余弦退火学习率调度器"""

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # 修复：确保第一个epoch就有合理的学习率
            lr = self.base_lr * max(1, epoch) / self.warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


def advanced_train(scene_idx=1, round_idx=1):
    """改进的训练函数"""
    print(f"Advanced Training - Scene {scene_idx} of Round {round_idx}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据路径
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    train_data_path = f"{data_dir}/Round{round_idx}TrainData{scene_idx}.npy"
    train_label_path = f"{data_dir}/Round{round_idx}TrainLabel{scene_idx}.npy"

    # 检查文件是否存在
    if not all(os.path.exists(f) for f in [cfg_path, train_data_path, train_label_path]):
        print("Some data files are missing!")
        return None

    # 读取配置
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    print(f"Config: samp_num={samp_num}, M={M}, N={N}, R={R}")

    # 加载数据
    print("Loading training data...")
    train_data = np.load(train_data_path)  # [N_samp, M, N, 2]
    train_label = np.load(train_label_path)  # [N_samp, M, N, 2]

    print(f"Data shapes: data={train_data.shape}, label={train_label.shape}")

    # 数据预处理和归一化
    print("Preprocessing data...")

    # 计算数据统计信息
    data_mean = np.mean(train_data, axis=(0, 1, 2), keepdims=True)
    data_std = np.std(train_data, axis=(0, 1, 2), keepdims=True) + 1e-8

    label_mean = np.mean(train_label, axis=(0, 1, 2), keepdims=True)
    label_std = np.std(train_label, axis=(0, 1, 2), keepdims=True) + 1e-8

    # 归一化
    train_data = (train_data - data_mean) / data_std
    train_label = (train_label - label_mean) / label_std

    # 数据分割为训练集和验证集
    val_ratio = 0.1
    val_size = int(samp_num * val_ratio)
    indices = np.random.permutation(samp_num)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_data_split = train_data[train_indices]
    train_label_split = train_label[train_indices]
    val_data = train_data[val_indices]
    val_label = train_label[val_indices]

    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")

    # 创建模型
    model = SVDNet(M=M, N=N, R=R).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # 训练参数
    num_epochs = 80
    batch_size = 4  # 减小批次大小以适应大模型
    warmup_epochs = 5

    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, num_epochs, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=15, min_delta=1e-6)

    print(f"Starting training for {num_epochs} epochs...")

    # 训练循环
    best_val_loss = float('inf')
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 更新学习率
        current_lr = scheduler.step(epoch)

        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        epoch_train_ae = 0.0
        num_train_batches = 0

        # 随机打乱训练数据
        train_perm = np.random.permutation(len(train_indices))

        pbar = tqdm(range(0, len(train_indices), batch_size),
                    desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(train_indices))
            batch_indices = train_perm[batch_start:batch_end]

            batch_loss = 0.0
            batch_ae = 0.0

            optimizer.zero_grad()

            # 处理批次中的每个样本
            for idx in batch_indices:
                # 获取单个样本
                H_data = torch.FloatTensor(train_data_split[idx]).to(device)
                H_label = torch.FloatTensor(train_label_split[idx]).to(device)

                # 前向传播
                U_out, S_out, V_out = model(H_data)

                # 动态调整损失权重
                ortho_weight = 0.1 + 0.4 * (epoch / num_epochs)  # 逐渐增加正交性约束
                singular_weight = 0.05

                # 计算损失
                loss, recon_loss, U_ortho_loss, V_ortho_loss, singular_loss = compute_loss(
                    U_out, S_out, V_out, H_label,
                    lambda_ortho=ortho_weight,
                    lambda_singular=singular_weight
                )

                batch_loss += loss

                # 计算AE指标
                ae = compute_ae_metric(U_out, S_out, V_out, H_label)
                batch_ae += ae

            # 平均损失
            batch_loss = batch_loss / len(batch_indices)
            batch_ae = batch_ae / len(batch_indices)

            # 反向传播
            batch_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_train_loss += batch_loss.item()
            epoch_train_ae += batch_ae
            num_train_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}',
                'AE': f'{batch_ae:.4f}',
                'LR': f'{current_lr:.1e}'
            })

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_ae = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_start in range(0, len(val_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(val_indices))

                batch_val_loss = 0.0
                batch_val_ae = 0.0

                for idx in range(batch_start, batch_end):
                    H_data = torch.FloatTensor(val_data[idx]).to(device)
                    H_label = torch.FloatTensor(val_label[idx]).to(device)

                    U_out, S_out, V_out = model(H_data)

                    loss, _, _, _, _ = compute_loss(
                        U_out, S_out, V_out, H_label,
                        lambda_ortho=ortho_weight,
                        lambda_singular=singular_weight
                    )

                    ae = compute_ae_metric(U_out, S_out, V_out, H_label)

                    batch_val_loss += loss.item()
                    batch_val_ae += ae

                batch_val_loss /= (batch_end - batch_start)
                batch_val_ae /= (batch_end - batch_start)

                epoch_val_loss += batch_val_loss
                epoch_val_ae += batch_val_ae
                num_val_batches += 1

        # 计算平均指标
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_ae = epoch_train_ae / num_train_batches
        avg_val_loss = epoch_val_loss / num_val_batches
        avg_val_ae = epoch_val_ae / num_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, AE: {avg_train_ae:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, AE: {avg_val_ae:.4f}")
        print(f"  LR: {current_lr:.1e}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
            torch.save(best_model_state, model_path)
            print(f"  ✓ Best model saved: {model_path}")

        # 早停检查
        if early_stopping(avg_val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Training completed! Best val loss: {best_val_loss:.4f}")

    # 保存训练统计信息
    stats = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'data_stats': {
            'data_mean': data_mean,
            'data_std': data_std,
            'label_mean': label_mean,
            'label_std': label_std
        }
    }

    stats_path = f"training_stats_round{round_idx}_scene{scene_idx}.npz"
    np.savez(stats_path, **stats)
    print(f"Training statistics saved: {stats_path}")

    return model, stats


def train_all_scenes_advanced(round_idx=1):
    """训练所有场景 - 改进版"""
    scenes = [1, 2, 3]
    all_results = {}

    for scene_idx in scenes:
        print(f"\n{'=' * 60}")
        print(f"Training Scene {scene_idx}")
        print(f"{'=' * 60}")

        try:
            model, stats = advanced_train(scene_idx, round_idx)
            if model is not None:
                all_results[scene_idx] = {
                    'model': model,
                    'stats': stats
                }
                print(f"✓ Scene {scene_idx} training completed successfully")
                print(f"  Best validation loss: {stats['best_val_loss']:.4f}")
            else:
                print(f"✗ Scene {scene_idx} training failed")
        except Exception as e:
            print(f"✗ Error training scene {scene_idx}: {e}")
            continue

    print("\nAll scenes training completed!")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Advanced SVD Training')
    parser.add_argument('--scene', type=int, default=None, help='Scene number (if None, train all)')
    parser.add_argument('--round', type=int, default=1, help='Round number')

    args = parser.parse_args()

    if args.scene is not None:
        advanced_train(args.scene, args.round)
    else:
        train_all_scenes_advanced(args.round)