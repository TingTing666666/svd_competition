import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FastOrthogonal(nn.Module):
    def __init__(self, iterations=1):  # 减少迭代次数
        super().__init__()
        self.iterations = iterations

    def forward(self, x):
        # x: [M, R, 2]
        M, R, _ = x.shape
        x_complex = x[..., 0] + 1j * x[..., 1]

        # 简化的正交化 - 只做归一化和一次修正
        # 首先列归一化
        norms = torch.sqrt(torch.sum(torch.abs(x_complex) ** 2, dim=0, keepdim=True) + 1e-10)
        Q = x_complex / norms

        # 单次正交化修正
        if self.iterations > 0:
            gram = Q.conj().T @ Q
            I = torch.eye(R, device=Q.device, dtype=Q.dtype)
            correction = Q @ (gram - I) * 0.5
            Q = Q - correction

            # 重新归一化
            norms = torch.sqrt(torch.sum(torch.abs(Q) ** 2, dim=0, keepdim=True) + 1e-10)
            Q = Q / norms

        return torch.stack([Q.real, Q.imag], dim=-1)


class LightweightSVDNet(nn.Module):
    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # 轻量级特征提取器
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2),  # 减少通道数，增加步长
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 简化的特征处理
        self.feature_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 奇异值预测
        self.s_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, R),
            nn.Softplus()
        )

        # U矩阵预测
        self.u_net = nn.Sequential(
            nn.Linear(256 + R, 512),
            nn.ReLU(),
            nn.Linear(512, M * R * 2)
        )

        # V矩阵预测
        self.v_net = nn.Sequential(
            nn.Linear(256 + R, 512),
            nn.ReLU(),
            nn.Linear(512, N * R * 2)
        )

        self.orthogonal = FastOrthogonal(iterations=1)
        self.s_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: [M, N, 2]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [M, N, 2], got {x.dim()}D with shape {x.shape}")

        x_conv = x.permute(2, 0, 1).unsqueeze(0)  # [1, 2, M, N]

        features = self.backbone(x_conv).squeeze(0)
        features = self.feature_net(features)

        # 预测奇异值
        s_raw = self.s_net(features)
        s_values = torch.sort(s_raw * torch.abs(self.s_scale), descending=True)[0]

        # 预测U和V
        context = torch.cat([features, s_values.detach()], dim=-1)

        U_flat = self.u_net(context)
        V_flat = self.v_net(context)

        U = U_flat.view(self.M, self.R, 2)
        V = V_flat.view(self.N, self.R, 2)

        # 快速正交化
        U = self.orthogonal(U)
        V = self.orthogonal(V)

        return U, s_values, V


def fast_compute_loss(U, S, V, H_ideal, lambda_ortho=0.5):
    # 简化的损失函数
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # SVD重构
    H_recon = torch.einsum('mr,r,nr->mn', U_complex, S, V_complex.conj())

    # 重构损失 - 只用Frobenius范数
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro')
    ideal_norm = torch.norm(H_ideal_complex, p='fro')
    recon_loss = recon_error / (ideal_norm + 1e-8)

    # 简化的正交性约束
    R = S.shape[0]
    I = torch.eye(R, device=U.device, dtype=torch.complex64)

    U_gram = U_complex.conj().T @ U_complex
    V_gram = V_complex.conj().T @ V_complex

    U_ortho_loss = torch.norm(U_gram - I, p='fro')
    V_ortho_loss = torch.norm(V_gram - I, p='fro')

    # 奇异值约束
    monotonic_loss = torch.sum(F.relu(S[1:] - S[:-1] + 1e-6)) if R > 1 else torch.tensor(0.0, device=S.device)
    positive_loss = torch.sum(F.relu(1e-6 - S))

    total_loss = (recon_loss +
                  lambda_ortho * (U_ortho_loss + V_ortho_loss) +
                  0.01 * (monotonic_loss + positive_loss))

    return total_loss, recon_loss, U_ortho_loss, V_ortho_loss


def compute_ae_metric(U, S, V, H_ideal):
    # 计算AE指标
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    H_recon = torch.einsum('mr,r,nr->mn', U_complex, S, V_complex.conj())

    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro') / torch.norm(H_ideal_complex, p='fro')

    I = torch.eye(S.shape[0], device=U.device, dtype=torch.complex64)
    U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    ae = recon_error + U_ortho_error + V_ortho_error
    return ae.item()


# 保持兼容性
SVDNet = LightweightSVDNet
compute_loss = fast_compute_loss