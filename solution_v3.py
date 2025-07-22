import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MinimalOrthogonal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        M, R, _ = x.shape
        x_complex = x[..., 0] + 1j * x[..., 1]

        # 快速正交化
        Q = torch.zeros_like(x_complex)
        for j in range(R):
            v = x_complex[:, j]
            for i in range(j):
                proj = torch.sum(torch.conj(Q[:, i]) * v)
                v = v - proj * Q[:, i]
            norm = torch.sqrt(torch.sum(torch.abs(v) ** 2) + 1e-8)
            Q = Q.clone()
            Q[:, j] = v / norm

        return torch.stack([Q.real, Q.imag], dim=-1)


class SVDNet(nn.Module):
    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # 极简网络
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # 直接预测
        self.s_net = nn.Linear(256, R)
        self.u_net = nn.Linear(256, M * R * 2)
        self.v_net = nn.Linear(256, N * R * 2)

        self.orthogonal = MinimalOrthogonal()

    def forward(self, x):
        x_conv = x.permute(2, 0, 1).unsqueeze(0)
        features = self.encoder(x_conv).squeeze(0)

        # 奇异值 - 简单激活
        s_values = F.softplus(self.s_net(features))
        s_values = torch.sort(s_values, descending=True)[0]

        # U和V矩阵
        u_matrix = self.u_net(features).view(self.M, self.R, 2)
        v_matrix = self.v_net(features).view(self.N, self.R, 2)

        # 正交化
        u_matrix = self.orthogonal(u_matrix)
        v_matrix = self.orthogonal(v_matrix)

        return u_matrix, s_values, v_matrix


def compute_loss(U, S, V, H_ideal, lambda_ortho=0.3):
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    S_diag = torch.diag(S.to(torch.complex64))
    H_recon = U_complex @ S_diag @ V_complex.conj().T

    recon_loss = torch.norm(H_ideal_complex - H_recon, p='fro') / (torch.norm(H_ideal_complex, p='fro') + 1e-8)

    R = S.shape[0]
    I = torch.eye(R, device=U.device, dtype=torch.complex64)

    U_ortho_loss = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_loss = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    total_loss = recon_loss + lambda_ortho * (U_ortho_loss + V_ortho_loss)

    return total_loss, recon_loss, U_ortho_loss, V_ortho_loss


def compute_ae_metric(U, S, V, H_ideal):
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    S_diag = torch.diag(S.to(torch.complex64))
    H_recon = U_complex @ S_diag @ V_complex.conj().T

    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro') / torch.norm(H_ideal_complex, p='fro')

    I = torch.eye(S.shape[0], device=U.device, dtype=torch.complex64)
    U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    ae = recon_error + U_ortho_error + V_ortho_error
    return ae.item()