import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from solution import SVDNet, compute_loss, compute_ae_metric

def read_cfg_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
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
    """
    归一化复数张量 H，使得每个样本的 Frobenius 范数 = 1。
    H: [B, M, N, 2] 或 [M, N, 2]
    """
    if H.dim() == 4:
        B, M, N, _ = H.shape
        flat = H.view(B, -1, 2)
        # 每个样本的平方和
        sq = flat[...,0]**2 + flat[...,1]**2  # [B, M*N]
        fro = torch.sqrt(sq.sum(dim=1, keepdim=True))  # [B,1]
        return (flat / (fro.unsqueeze(-1) + 1e-8)).view_as(H)
    else:
        # 单样本模式
        flat = H.view(-1, 2)
        sq = flat[...,0]**2 + flat[...,1]**2
        fro = torch.sqrt(sq.sum())
        return H / (fro + 1e-8)

def simple_train(scene_idx=1, round_idx=1):
    print(">>> Enter simple_train")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = f"./CompetitionData{round_idx}"
    cfg_path  = f"{data_dir}/Round{round_idx}CfgData{scene_idx}.txt"
    data_path = f"{data_dir}/Round{round_idx}TrainData{scene_idx}.npy"
    lab_path  = f"{data_dir}/Round{round_idx}TrainLabel{scene_idx}.npy"

    print("cfg_path =", cfg_path)
    print("data_path =", data_path)
    print("lab_path  =", lab_path)

    if not all(os.path.exists(p) for p in [cfg_path, data_path, lab_path]):
        print("Some data files are missing!")
        return

    samp, M, N, IQ, R = read_cfg_file(cfg_path)
    print("cfg =", samp, M, N, IQ, R)

    train_data  = np.load(data_path).astype(np.float32)
    train_label = np.load(lab_path).astype(np.float32)
    print("data shapes:", train_data.shape, train_label.shape)

    ds = HDataset(train_data, train_label)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0, drop_last=False)

    model = SVDNet(M=M, N=N, R=R).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_loss = float('inf')
    epochs = 30

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_ae   = 0.0
        batches    = 0

        pbar = tqdm(dl, desc=f"S{scene_idx} Ep{ep+1}/{epochs}")
        for H_in_np, H_lab_np in pbar:
            # 转 tensor + 放设备
            H_in  = H_in_np.to(device)
            H_lab = H_lab_np.to(device)

            # 统一归一化
            H_in  = normalize(H_in)
            H_lab = normalize(H_lab)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                U_b, S_b, V_b = model(H_in)   # 支持 batch forward
                loss, rec, uo, vo = compute_loss(U_b, S_b, V_b, H_lab, lambda_ortho=0.3)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                ae = compute_ae_metric(U_b, S_b, V_b, H_lab)

            total_loss += loss.item()
            total_ae   += ae
            batches    += 1
            pbar.set_postfix(L=f"{loss.item():.4f}", R=f"{rec.item():.4f}")

        avg_loss = total_loss / batches
        avg_ae   = total_ae   / batches
        print(f"Epoch {ep+1}: Loss={avg_loss:.4f}, AE={avg_ae:.4f}")

        # 保存最佳
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = f"svd_model_round{round_idx}_scene{scene_idx}.pth"
            torch.save(model.state_dict(), ckpt)
            print("  Saved best:", ckpt)

        scheduler.step()

    print("Training done. Best loss:", best_loss)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--scene", type=int, default=1)
    args = parser.parse_args()
    simple_train(scene_idx=args.scene, round_idx=args.round)
