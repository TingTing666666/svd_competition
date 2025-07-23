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

    # Data paths
    data_dir = f"./CompetitionData{round_idx}"
    cfg_path = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    train_data_path = f"{data_dir}/Round{round_idx}TrainData{scene_idx}.npy"
    train_label_path = f"{data_dir}/Round{round_idx}TrainLabel{scene_idx}.npy"

    if not all(os.path.exists(f) for f in [cfg_path, train_data_path, train_label_path]):
        print(f"Missing data files for scene {scene_idx}")
        return None

    # Load config and data
    samp_num, M, N, IQ, R = read_cfg_file(cfg_path)
    train_data = np.load(train_data_path).astype(np.float32)
    train_label = np.load(train_label_path).astype(np.float32)

    # Create model
    model = SVDNet(M=M, N=N, R=R).to(device)

    # Optimizer with layered learning rates
    param_groups = [
        {'params': list(model.backbone.parameters()), 'lr': 5e-4, 'weight_decay': 1e-4},
        {'params': list(model.feature_net.parameters()), 'lr': 8e-4, 'weight_decay': 5e-5},
        {'params': list(model.s_net.parameters()) + list(model.u_net.parameters()) + list(model.v_net.parameters()),
         'lr': 1e-3, 'weight_decay': 1e-5},
        {'params': list(model.orthogonal.parameters()), 'lr': 1.5e-3, 'weight_decay': 0},
        {'params': [model.s_scale], 'lr': 2e-3, 'weight_decay': 0}
    ]

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    # Training config
    num_epochs = 30
    batch_size = 32
    best_ae = float('inf')

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_ae = 0.0
        num_batches = 0

        # Adaptive loss weights
        progress = epoch / num_epochs
        lambda_ortho = 0.8 + 0.6 * progress  # 0.8 -> 1.4
        lambda_energy = 0.05 + 0.05 * progress  # 0.05 -> 0.1

        indices = np.random.permutation(samp_num)

        for batch_start in tqdm(range(0, samp_num, batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch_end = min(batch_start + batch_size, samp_num)
            batch_indices = indices[batch_start:batch_end]

            batch_loss = 0.0
            batch_ae = 0.0

            optimizer.zero_grad()

            for idx in batch_indices:
                H_data = torch.FloatTensor(train_data[idx]).to(device)
                H_label = torch.FloatTensor(train_label[idx]).to(device)

                U_out, S_out, V_out = model(H_data)

                loss, recon_loss, U_ortho_loss, V_ortho_loss = compute_loss(
                    U_out, S_out, V_out, H_label,
                    lambda_ortho=lambda_ortho, lambda_energy=lambda_energy
                )

                batch_loss += loss
                ae = compute_ae_metric(U_out, S_out, V_out, H_label)
                batch_ae += ae

            batch_loss = batch_loss / len(batch_indices)
            batch_ae = batch_ae / len(batch_indices)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_ae += batch_ae
            num_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_ae = epoch_ae / num_batches

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.5f}, AE={avg_ae:.5f}")

        if avg_ae < best_ae:
            best_ae = avg_ae
            model_path = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
            torch.save(model.state_dict(), model_path)

    print(f"Scene {scene_idx} completed. Best AE: {best_ae:.5f}")
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