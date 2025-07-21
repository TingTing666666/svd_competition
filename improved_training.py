import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
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


def improved_train(scene_idx=1, round_idx=1):
    """改进的训练函数 - 符合比赛要求"""
    print(f"🚀 Training Scene {scene_idx} of Round {round_idx}")

    # 开启梯度异常检测（调试用）
    torch.autograd.set_detect_anomaly(True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")

    # 数据路径
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    train_data_path = f"{data_dir}/Round{round_idx}TrainData{scene_idx}.npy"
    train_label_path = f"{data_dir}/Round{round_idx}TrainLabel{scene_idx}.npy"

    # 检查文件是否存在
    if not all(os.path.exists(f) for f in [cfg_path, train_data_path, train_label_path]):
        print("❌ Some data files are missing!")
        return None

    # 读取配置
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    print(f"📊 Config: samp_num={samp_num}, M={M}, N={N}, R={R}")

    # 加载数据
    print("📂 Loading training data...")
    train_data = np.load(train_data_path)  # [N_samp, M, N, 2]
    train_label = np.load(train_label_path)  # [N_samp, M, N, 2]

    print(f"📈 Data shapes: data={train_data.shape}, label={train_label.shape}")

    # 数据划分
    val_ratio = 0.1
    val_size = int(samp_num * val_ratio)
    indices = np.random.permutation(samp_num)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    print(f"🔄 Train: {len(train_indices)}, Val: {len(val_indices)}")

    # 创建模型
    model = SVDNet(M=M, N=N, R=R).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {total_params:,}")

    # 训练参数 - 在这里定义！
    num_epochs = 60
    batch_size = 8
    best_val_loss = float('inf')

    # 优化器 - 使用更保守的学习率
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # 学习率调度器 - 使用余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    print(f"🏋️ Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        train_aes = []

        # 打乱训练数据
        np.random.shuffle(train_indices)

        pbar = tqdm(range(0, len(train_indices), batch_size),
                    desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(train_indices))
            batch_indices = train_indices[batch_start:batch_end]

            optimizer.zero_grad()
            batch_loss = 0.0
            batch_ae = 0.0

            for idx in batch_indices:
                H_data = torch.FloatTensor(train_data[idx]).to(device)  # [M, N, 2]
                H_label = torch.FloatTensor(train_label[idx]).to(device)  # [M, N, 2]

                # 前向传播
                U_out, S_out, V_out = model(H_data)

                # 计算损失 - 使用修复后的compute_loss
                loss, recon_loss, U_ortho_loss, V_ortho_loss = compute_loss(
                    U_out, S_out, V_out, H_label,
                    lambda_ortho=0.5,  # 降低正交性权重
                    lambda_recon=1.0  # 保持重构权重
                )

                # 累积损失
                batch_loss = batch_loss + loss

                # 计算AE指标（使用detach避免梯度问题）
                with torch.no_grad():
                    ae = compute_ae_metric(U_out.detach(), S_out.detach(), V_out.detach(), H_label)
                    batch_ae += ae

            # 平均损失
            batch_loss = batch_loss / len(batch_indices)
            batch_ae = batch_ae / len(batch_indices)

            # 反向传播
            batch_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(batch_loss.item())
            train_aes.append(batch_ae)

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}',
                'AE': f'{batch_ae:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })

        # 验证阶段
        model.eval()
        val_losses = []
        val_aes = []

        with torch.no_grad():
            for idx in val_indices[:min(200, len(val_indices))]:  # 限制验证样本
                H_data = torch.FloatTensor(train_data[idx]).to(device)
                H_label = torch.FloatTensor(train_label[idx]).to(device)

                U_out, S_out, V_out = model(H_data)
                loss, _, _, _ = compute_loss(U_out, S_out, V_out, H_label, lambda_ortho=0.5)
                ae = compute_ae_metric(U_out, S_out, V_out, H_label)

                val_losses.append(loss.item())
                val_aes.append(ae)

        # 学习率调度
        scheduler.step()

        # 计算平均指标
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_ae = np.mean(train_aes)
        avg_val_ae = np.mean(val_aes)

        print(f"📊 Epoch {epoch + 1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        print(f"   Train AE={avg_train_ae:.4f}, Val AE={avg_val_ae:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            timestamp = datetime.now().strftime("%y_%m_%d_%H_%M")

            # 保存详细checkpoint
            detailed_path = f"svd_model_round{round_idx}_scene{scene_idx}_{timestamp}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'config': {'M': M, 'N': N, 'R': R}
            }, detailed_path)

            # 保存标准格式（用于测试）
            standard_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
            torch.save(model.state_dict(), standard_path)

            print(f"💾 Best model saved: {standard_path}")

        # 早停检查
        if epoch > 20 and avg_val_loss > best_val_loss * 1.1:
            patience_counter = getattr(improved_train, 'patience_counter', 0) + 1
            if patience_counter >= 10:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break
        else:
            improved_train.patience_counter = 0

    print(f"🎉 Training completed! Best val loss: {best_val_loss:.4f}")

    # 关闭梯度异常检测
    torch.autograd.set_detect_anomaly(False)

    return model


def train_all_scenes_improved(round_idx=1):
    """训练所有场景 - 改进版"""
    scenes = [1, 2, 3]

    print("🏆" * 20)
    print(f"IMPROVED TRAINING - ROUND {round_idx}")
    print("🏆" * 20)

    for scene_idx in scenes:
        print(f"\n{'=' * 50}")
        print(f"🎯 Training Scene {scene_idx}")
        print(f"{'=' * 50}")

        try:
            model = improved_train(scene_idx, round_idx)
            if model is not None:
                print(f"✅ Scene {scene_idx} completed successfully")
            else:
                print(f"❌ Scene {scene_idx} failed")
        except Exception as e:
            print(f"💥 Error training scene {scene_idx}: {e}")
            continue

    print("\n🎉 All scenes training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Improved SVD Training')
    parser.add_argument('--scene', type=int, default=None, help='Scene number (if None, train all)')
    parser.add_argument('--round', type=int, default=1, help='Round number')

    args = parser.parse_args()

    if args.scene is not None:
        improved_train(args.scene, args.round)
    else:
        train_all_scenes_improved(args.round)