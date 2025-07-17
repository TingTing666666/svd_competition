import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
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


def simple_train(scene_idx=1, round_idx=1):
    """简单训练函数"""
    print(f"Training Scene {scene_idx} of Round {round_idx}")

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

    # 创建模型
    model = SVDNet(M=M, N=N, R=R).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 训练参数
    num_epochs = 30
    batch_size = 8  # 小批量以节省内存

    print(f"Starting training for {num_epochs} epochs...")

    # 训练循环
    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_ae = 0.0
        num_batches = 0

        # 随机打乱数据
        indices = np.random.permutation(samp_num)

        pbar = tqdm(range(0, samp_num, batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, samp_num)
            batch_indices = indices[batch_start:batch_end]

            batch_loss = 0.0
            batch_ae = 0.0

            optimizer.zero_grad()

            # 处理批次中的每个样本
            for idx in batch_indices:
                # 获取单个样本
                H_data = torch.FloatTensor(train_data[idx]).to(device)  # [M, N, 2]
                H_label = torch.FloatTensor(train_label[idx]).to(device)  # [M, N, 2]

                # 前向传播
                U_out, S_out, V_out = model(H_data)

                # 计算损失
                loss, recon_loss, U_ortho_loss, V_ortho_loss = compute_loss(
                    U_out, S_out, V_out, H_label, lambda_ortho=0.1
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

            epoch_loss += batch_loss.item()
            epoch_ae += batch_ae
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}',
                'AE': f'{batch_ae:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })

        # 学习率调度
        scheduler.step()

        # 计算平均指标
        avg_loss = epoch_loss / num_batches
        avg_ae = epoch_ae / num_batches

        print(f"Epoch {epoch + 1}/{num_epochs}: Loss={avg_loss:.4f}, AE={avg_ae:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved: {model_path}")

    print(f"Training completed! Best loss: {best_loss:.4f}")
    return model


def train_all_scenes(round_idx=1):
    """训练所有场景"""
    scenes = [1, 2, 3]

    for scene_idx in scenes:
        print(f"\n{'=' * 50}")
        print(f"Training Scene {scene_idx}")
        print(f"{'=' * 50}")

        try:
            model = simple_train(scene_idx, round_idx)
            if model is not None:
                print(f"✓ Scene {scene_idx} training completed successfully")
            else:
                print(f"✗ Scene {scene_idx} training failed")
        except Exception as e:
            print(f"✗ Error training scene {scene_idx}: {e}")
            continue

    print("\nAll scenes training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Simple SVD Training')
    parser.add_argument('--scene', type=int, default=None, help='Scene number (if None, train all)')
    parser.add_argument('--round', type=int, default=1, help='Round number')

    args = parser.parse_args()

    if args.scene is not None:
        simple_train(args.scene, args.round)
    else:
        train_all_scenes(args.round)