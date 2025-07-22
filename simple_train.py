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


def validate_model(model, val_data, val_label, device, max_samples=200):
    model.eval()
    val_losses = []
    val_aes = []

    val_samples = min(max_samples, val_data.shape[0])
    val_indices = np.random.choice(val_data.shape[0], val_samples, replace=False)

    with torch.no_grad():
        for idx in val_indices:
            H_data = torch.FloatTensor(val_data[idx]).to(device)
            H_label = torch.FloatTensor(val_label[idx]).to(device)

            U_out, S_out, V_out = model(H_data)

            loss, _, _, _ = compute_loss(U_out, S_out, V_out, H_label)
            ae = compute_ae_metric(U_out, S_out, V_out, H_label)

            val_losses.append(loss.item())
            val_aes.append(ae)

    return np.mean(val_losses), np.mean(val_aes)


def simple_train(scene_idx=1, round_idx=1):
    print(f"Training Scene {scene_idx} of Round {round_idx}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 数据路径
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    train_data_path = f"{data_dir}/Round{round_idx}TrainData{scene_idx}.npy"
    train_label_path = f"{data_dir}/Round{round_idx}TrainLabel{scene_idx}.npy"

    if not all(os.path.exists(f) for f in [cfg_path, train_data_path, train_label_path]):
        print("Data files missing!")
        return None

    # 读取配置
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    print(f"Config: samp_num={samp_num}, M={M}, N={N}, R={R}")

    # 加载数据
    train_data = np.load(train_data_path).astype(np.float32)
    train_label = np.load(train_label_path).astype(np.float32)
    print(f"Data loaded: {train_data.shape}")

    # 数据采样 - 使用部分数据快速训练
    sample_size = min(8000, samp_num)
    indices = np.random.choice(samp_num, sample_size, replace=False)
    train_data = train_data[indices]
    train_label = train_label[indices]

    # 划分训练集和验证集
    val_size = int(sample_size * 0.15)
    val_indices = np.random.choice(sample_size, val_size, replace=False)
    train_indices = np.setdiff1d(np.arange(sample_size), val_indices)

    val_data = train_data[val_indices]
    val_label = train_label[val_indices]
    train_data = train_data[train_indices]
    train_label = train_label[train_indices]

    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # 创建模型
    model = SVDNet(M=M, N=N, R=R).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # 训练参数
    num_epochs = 50
    batch_size = 32

    print(f"Training: {num_epochs} epochs, batch_size={batch_size}")

    # 训练循环
    best_val_ae = float('inf')
    patience = 12
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        epoch_ae = 0.0
        num_batches = 0

        # 打乱训练数据
        train_indices_epoch = np.random.permutation(len(train_data))

        pbar = tqdm(range(0, len(train_data), batch_size),
                    desc=f"Epoch {epoch + 1:2d}/{num_epochs}", leave=False)

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(train_data))
            batch_indices = train_indices_epoch[batch_start:batch_end]

            optimizer.zero_grad()

            batch_loss = 0.0
            batch_ae = 0.0

            for idx in batch_indices:
                H_data = torch.FloatTensor(train_data[idx]).to(device)
                H_label = torch.FloatTensor(train_label[idx]).to(device)

                # 前向传播
                U_out, S_out, V_out = model(H_data)

                # 计算损失
                loss, recon_loss, U_ortho_loss, V_ortho_loss = compute_loss(
                    U_out, S_out, V_out, H_label, lambda_ortho=0.4
                )

                loss = loss / len(batch_indices)
                loss.backward()

                batch_loss += loss.item() * len(batch_indices)

                # 计算AE
                with torch.no_grad():
                    ae = compute_ae_metric(U_out, S_out, V_out, H_label)
                    batch_ae += ae

            # 梯度更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_ae = batch_ae / len(batch_indices)

            epoch_loss += batch_loss
            epoch_ae += batch_ae
            num_batches += 1

            pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'AE': f'{batch_ae:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })

        scheduler.step()

        # 验证
        val_loss, val_ae = validate_model(model, val_data, val_label, device)

        avg_train_loss = epoch_loss / num_batches
        avg_train_ae = epoch_ae / num_batches

        print(f"Epoch {epoch + 1:2d}/{num_epochs}: "
              f"Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Train AE={avg_train_ae:.4f}, Val AE={val_ae:.4f}")

        # 保存最佳模型
        if val_ae < best_val_ae:
            best_val_ae = val_ae
            patience_counter = 0

            model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved: Val AE={best_val_ae:.4f}")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Training completed! Best Val AE: {best_val_ae:.4f}")
    return model


def train_all_scenes(round_idx=1):
    scenes = [1, 2, 3]

    for scene_idx in scenes:
        print(f"\nTraining Scene {scene_idx}")
        print("=" * 50)

        try:
            model = simple_train(scene_idx, round_idx)
            if model is not None:
                print(f"Scene {scene_idx} training completed successfully")
            else:
                print(f"Scene {scene_idx} training failed")
        except Exception as e:
            print(f"Error training scene {scene_idx}: {e}")
            continue

    print("\nAll scenes training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='SVD Training')
    parser.add_argument('--scene', type=int, default=None, help='Scene number (if None, train all)')
    parser.add_argument('--round', type=int, default=1, help='Round number')

    args = parser.parse_args()

    if args.scene is not None:
        simple_train(args.scene, args.round)
    else:
        train_all_scenes(args.round)