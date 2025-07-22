import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import importlib


def read_cfg_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    samp_num = int(lines[0].strip())
    M = int(lines[1].strip())
    N = int(lines[2].strip())
    IQ = int(lines[3].strip())
    R = int(lines[4].strip())
    return samp_num, M, N, IQ, R


def validate_model(model, val_data, val_label, device, compute_loss, compute_ae_metric, max_samples=100):
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


def train_version(version_name, scene_idx=1, round_idx=1):
    print(f"Training {version_name} - Scene {scene_idx} of Round {round_idx}")

    # 动态导入对应版本
    try:
        solution_module = importlib.import_module(f'solution_{version_name}')
        SVDNet = solution_module.SVDNet
        compute_loss = solution_module.compute_loss
        compute_ae_metric = solution_module.compute_ae_metric
    except ImportError:
        print(f"Cannot import {version_name}, skipping...")
        return None

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

    # 数据采样
    sample_size = min(6000 if version_name == 'v3' else 8000, samp_num)
    indices = np.random.choice(samp_num, sample_size, replace=False)
    train_data = train_data[indices]
    train_label = train_label[indices]

    # 划分数据
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

    # 不同版本的训练配置
    if version_name == 'v1':  # 强正交化版本
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        num_epochs = 60
        batch_size = 16
    elif version_name == 'v2':  # 注意力版本
        optimizer = optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=5e-5)
        num_epochs = 45
        batch_size = 24
    else:  # v3 轻量版本
        optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
        num_epochs = 40
        batch_size = 32

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    print(f"Training: {num_epochs} epochs, batch_size={batch_size}")

    # 训练循环
    best_val_ae = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0.0
        epoch_ae = 0.0
        num_batches = 0

        train_indices_epoch = np.random.permutation(len(train_data))

        pbar = tqdm(range(0, len(train_data), batch_size),
                    desc=f"{version_name} Epoch {epoch + 1:2d}/{num_epochs}", leave=False)

        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, len(train_data))
            batch_indices = train_indices_epoch[batch_start:batch_end]

            optimizer.zero_grad()

            batch_loss = 0.0
            batch_ae = 0.0

            for idx in batch_indices:
                H_data = torch.FloatTensor(train_data[idx]).to(device)
                H_label = torch.FloatTensor(train_label[idx]).to(device)

                U_out, S_out, V_out = model(H_data)

                loss, _, _, _ = compute_loss(U_out, S_out, V_out, H_label)

                loss = loss / len(batch_indices)
                loss.backward()

                batch_loss += loss.item() * len(batch_indices)

                with torch.no_grad():
                    ae = compute_ae_metric(U_out, S_out, V_out, H_label)
                    batch_ae += ae

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_ae = batch_ae / len(batch_indices)

            epoch_loss += batch_loss
            epoch_ae += batch_ae
            num_batches += 1

            pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'AE': f'{batch_ae:.4f}'
            })

        scheduler.step()

        # 验证
        val_loss, val_ae = validate_model(model, val_data, val_label, device,
                                          compute_loss, compute_ae_metric)

        avg_train_loss = epoch_loss / num_batches
        avg_train_ae = epoch_ae / num_batches

        print(f"{version_name} Epoch {epoch + 1:2d}/{num_epochs}: "
              f"Train AE={avg_train_ae:.4f}, Val AE={val_ae:.4f}")

        # 保存最佳模型
        if val_ae < best_val_ae:
            best_val_ae = val_ae
            patience_counter = 0

            model_path = f"svd_model_{version_name}_round{round_idx}_scene{scene_idx}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"{version_name} Best model saved: Val AE={best_val_ae:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"{version_name} Early stopping at epoch {epoch + 1}")
            break

    print(f"{version_name} Training completed! Best Val AE: {best_val_ae:.4f}")
    return best_val_ae


def train_all_versions(scene_idx=1, round_idx=1):
    versions = ['v1', 'v2', 'v3']
    results = {}

    print(f"Training All Versions - Scene {scene_idx} of Round {round_idx}")
    print("=" * 60)

    for version in versions:
        print(f"\n{'-' * 20} {version.upper()} {'-' * 20}")
        try:
            best_ae = train_version(version, scene_idx, round_idx)
            results[version] = best_ae
            print(f"{version} completed with AE: {best_ae:.4f}")
        except Exception as e:
            print(f"Error training {version}: {e}")
            results[version] = float('inf')

    # 找出最佳版本
    best_version = min(results.keys(), key=lambda v: results[v])
    print(f"\nBest version: {best_version} with AE: {results[best_version]:.4f}")

    # 复制最佳模型为标准命名
    best_model_path = f"svd_model_{best_version}_round{round_idx}_scene{scene_idx}.pth"
    standard_model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"

    if os.path.exists(best_model_path):
        import shutil
        shutil.copy2(best_model_path, standard_model_path)
        print(f"Best model copied to {standard_model_path}")

    return results


def train_all_scenes_all_versions(round_idx=1):
    scenes = [1, 2, 3]
    all_results = {}

    for scene_idx in scenes:
        print(f"\n{'=' * 80}")
        print(f"SCENE {scene_idx}")
        print(f"{'=' * 80}")

        scene_results = train_all_versions(scene_idx, round_idx)
        all_results[scene_idx] = scene_results

    # 总结
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")

    for scene_idx in scenes:
        print(f"Scene {scene_idx}:")
        for version, ae in all_results[scene_idx].items():
            print(f"  {version}: {ae:.4f}")
        best_v = min(all_results[scene_idx].keys(), key=lambda v: all_results[scene_idx][v])
        print(f"  Best: {best_v}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Version SVD Training')
    parser.add_argument('--version', type=str, default=None, choices=['v1', 'v2', 'v3'],
                        help='Specific version to train')
    parser.add_argument('--scene', type=int, default=None, help='Scene number')
    parser.add_argument('--round', type=int, default=1, help='Round number')
    parser.add_argument('--all', action='store_true', help='Train all versions for all scenes')

    args = parser.parse_args()

    if args.all:
        train_all_scenes_all_versions(args.round)
    elif args.version and args.scene:
        train_version(args.version, args.scene, args.round)
    elif args.scene:
        train_all_versions(args.scene, args.round)
    else:
        print("Please specify --version and --scene, or --scene, or --all")