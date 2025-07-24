# simple_train.py

import os
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from solution import SVDNet, compute_loss, compute_ae_metric

def read_cfg_file(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        nums = [l.strip() for l in f if l.strip()]
    return int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]), int(nums[4])

class HDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.data[i]), torch.from_numpy(self.label[i])

def normalize(H: torch.Tensor) -> torch.Tensor:
    # Frobenius 归一化：每个样本 ||H||_F = 1
    B, M, N, _ = H.shape
    flat = H.view(B, -1, 2)
    sq = flat[...,0]**2 + flat[...,1]**2
    fro = torch.sqrt(sq.sum(dim=1, keepdim=True))  # [B,1]
    return (flat / (fro.unsqueeze(-1) + 1e-8)).view_as(H)

def simple_train(round_idx: int, scene_idx: int):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取配置与数据
    data_dir = f"./CompetitionData{round_idx}"
    N_samp, M, N, IQ, R = read_cfg_file(
        os.path.join(data_dir, f"Round{round_idx}CfgData{scene_idx}.txt")
    )
    X = np.load(os.path.join(data_dir, f"Round{round_idx}TrainData{scene_idx}.npy")).astype(np.float32)
    Y = np.load(os.path.join(data_dir, f"Round{round_idx}TrainLabel{scene_idx}.npy")).astype(np.float32)
    print(f"Scene {scene_idx} data shape:", X.shape)

    ds = HDataset(X, Y)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

    # 构建轻量化模型
    model = SVDNet(M, N, R, hidden=96, num_blocks=2, ortho_iter=1).to(device)

    # 优化器 & 调度器
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    epochs_total = 40
    # CosineAnnealingLR over 40 epochs
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_total)

    scaler = torch.amp.GradScaler()

    # —— Stage 1: 冻结 U/V 头 ——
    for name, p in model.named_parameters():
        if name.startswith("head_U") or name.startswith("head_V"):
            p.requires_grad = False

    best_stage1 = float('inf')
    for ep in range(20):
        model.train()
        tot_loss = 0.0
        tot_ae   = 0.0
        nb = 0
        pbar = tqdm(dl, desc=f"Stage1 Ep{ep+1}/20")
        for x_np, y_np in pbar:
            Hx = normalize(x_np.to(device))
            Hy = normalize(y_np.to(device))

            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type=='cuda')):
                U, S, V = model(Hx)
                loss, rec, uo, vo = compute_loss(U, S, V, Hy, lambda_ortho=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                ae = compute_ae_metric(U, S, V, Hy)

            tot_loss += loss.item()
            tot_ae   += ae
            nb += 1
            pbar.set_postfix(L=f"{loss.item():.4f}", R=f"{rec.item():.4f}")

        avg_loss = tot_loss / nb
        avg_ae   = tot_ae   / nb
        print(f"[Stage1 Ep{ep+1}] Loss={avg_loss:.4f}, AE={avg_ae:.4f}")
        if avg_loss < best_stage1:
            best_stage1 = avg_loss
            torch.save(model.state_dict(), f"svd_stage1_r{round_idx}_s{scene_idx}.pth")
            print("  Saved best Stage1")
        sched.step()

    # —— Stage 2: 解冻全部参数 ——
    for p in model.parameters():
        p.requires_grad = True

    best_stage2 = float('inf')
    for ep in range(20):
        model.train()
        tot_loss = 0.0
        tot_ae   = 0.0
        nb = 0
        pbar = tqdm(dl, desc=f"Stage2 Ep{ep+1}/20")
        for x_np, y_np in pbar:
            Hx = normalize(x_np.to(device))
            Hy = normalize(y_np.to(device))

            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type=='cuda')):
                U, S, V = model(Hx)
                loss, rec, uo, vo = compute_loss(U, S, V, Hy, lambda_ortho=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                ae = compute_ae_metric(U, S, V, Hy)

            tot_loss += loss.item()
            tot_ae   += ae
            nb += 1
            pbar.set_postfix(L=f"{loss.item():.4f}", R=f"{rec.item():.4f}")

        avg_loss = tot_loss / nb
        avg_ae   = tot_ae   / nb
        print(f"[Stage2 Ep{ep+1}] Loss={avg_loss:.4f}, AE={avg_ae:.4f}")
        if avg_loss < best_stage2:
            best_stage2 = avg_loss
            torch.save(model.state_dict(), f"svd_stage2_r{round_idx}_s{scene_idx}.pth")
            print("  Saved best Stage2")
        sched.step()

    print("Training complete. Best Stage2 Loss:", best_stage2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--scene", type=int, default=1)
    args = parser.parse_args()
    simple_train(round_idx=args.round, scene_idx=args.scene)
