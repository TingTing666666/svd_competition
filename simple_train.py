import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from solution import SVDNet, compute_loss, compute_ae_metric


def read_cfg_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    samp_num = int(lines[0].strip())
    M = int(lines[1].strip())
    N = int(lines[2].strip())
    IQ = int(lines[3].strip())
    R = int(lines[4].strip())
    return samp_num, M, N, IQ, R


def train_scene(scene_idx=1, round_idx=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据路径
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    train_data_path = f"{data_dir}/Round{round_idx}TrainData{scene_idx}.npy"
    train_label_path = f"{data_dir}/Round{round_idx}TrainLabel{scene_idx}.npy"

    if not all(os.path.exists(f) for f in [cfg_path, train_data_path, train_label_path]):
        print(f"Missing data files for scene {scene_idx}")
        return None

    # 加载配置和数据
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    train_data = np.load(train_data_path).astype(np.float32)
    train_label = np.load(train_label_path).astype(np.float32)

    # 创建模型
    model = SVDNet(M=M, N=N, R=R).to(device)

    # 简化的优化器
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    # 快速训练配置
    num_epochs = 15  # 减少epoch数
    batch_size = 128  # 增大batch size
    best_ae = float('inf')

    # 数据子采样 - 只使用一部分数据加速训练
    if samp_num > 10000:
        subset_indices = np.random.choice(samp_num, 10000, replace=False)
        train_data = train_data[subset_indices]
        train_label = train_label[subset_indices]
        samp_num = 10000

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_ae = 0.0
        num_batches = 0

        # 自适应权重
        lambda_ortho = 0.3 + 0.4 * (epoch / num_epochs)  # 0.3 -> 0.7

        indices = np.random.permutation(samp_num)

        pbar = tqdm(range(0, samp_num, batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, samp_num)
            batch_indices = indices[batch_start:batch_end]

            optimizer.zero_grad()

            # 批量处理
            batch_losses = []
            batch_aes = []

            for idx in batch_indices:
                H_data = torch.FloatTensor(train_data[idx]).to(device)
                H_label = torch.FloatTensor(train_label[idx]).to(device)

                U_out, S_out, V_out = model(H_data)

                loss, _, _, _ = compute_loss(U_out, S_out, V_out, H_label, lambda_ortho=lambda_ortho)
                batch_losses.append(loss)

                # 每5个样本计算一次AE（减少计算）
                if len(batch_aes) < len(batch_losses) // 5:
                    ae = compute_ae_metric(U_out, S_out, V_out, H_label)
                    batch_aes.append(ae)

            # 批量损失
            if batch_losses:
                batch_loss = torch.stack(batch_losses).mean()
                batch_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += batch_loss.item()
                if batch_aes:
                    epoch_ae += np.mean(batch_aes)
                num_batches += 1

                # 更新进度条
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{batch_loss.item():.4f}',
                    'AE': f'{np.mean(batch_aes) if batch_aes else 0:.4f}',
                    'LR': f'{current_lr:.1e}'
                })

        scheduler.step()

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_ae = epoch_ae / num_batches if epoch_ae > 0 else float('inf')

            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, AE={avg_ae:.4f}")

            # 保存最佳模型
            if avg_ae < best_ae:
                best_ae = avg_ae
                model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
                torch.save(model.state_dict(), model_path)

    print(f"Scene {scene_idx} completed. Best AE: {best_ae:.4f}")
    return model


def train_all_scenes(round_idx=1):
    scenes = [1, 2, 3]
    for scene_idx in scenes:
        print(f"\nTraining Scene {scene_idx}")
        try:
            train_scene(scene_idx, round_idx)
        except Exception as e:
            print(f"Error training scene {scene_idx}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=int, default=None)
    parser.add_argument('--round', type=int, default=1)
    args = parser.parse_args()

    if args.scene is not None:
        train_scene(args.scene, args.round)
    else:
        train_all_scenes(args.round)