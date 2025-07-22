import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GramSchmidtLayer(nn.Module):
    """Gram-Schmidt正交化 - 不使用禁止算子"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [M, R, 2]
        M, R, _ = x.shape
        x_complex = x[..., 0] + 1j * x[..., 1]

        Q = torch.zeros_like(x_complex)

        for j in range(R):
            v = x_complex[:, j]

            for i in range(j):
                proj_coeff = torch.sum(torch.conj(Q[:, i]) * v)
                v = v - proj_coeff * Q[:, i]

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

        # 在你原有基础上加卷积层提升性能
        self.conv_features = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 保持你原有的网络结构
        self.feature_extractor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 奇异值预测网络
        self.singular_value_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, R),
            nn.Softplus()
        )

        # 左奇异矩阵预测网络
        self.left_singular_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, M * R * 2),
        )

        # 右奇异矩阵预测网络
        self.right_singular_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, N * R * 2),
        )

        # 替换禁止的QR分解
        self.orthogonal_layer = GramSchmidtLayer()

    def forward(self, x):
        # x: [M, N, 2]

        # 先用卷积提取空间特征
        x_conv = x.permute(2, 0, 1).unsqueeze(0)  # [1, 2, M, N]
        conv_feat = self.conv_features(x_conv).squeeze(0)  # [128]

        # 特征提取
        features = self.feature_extractor(conv_feat)  # [128]

        # 预测奇异值
        singular_values = self.singular_value_net(features)  # [R]
        singular_values = torch.sort(singular_values, dim=-1, descending=True)[0]

        # 预测左奇异矩阵
        U_flat = self.left_singular_net(features)  # [M*R*2]
        U = U_flat.view(self.M, self.R, 2)  # [M, R, 2]

        # 预测右奇异矩阵
        V_flat = self.right_singular_net(features)  # [N*R*2]
        V = V_flat.view(self.N, self.R, 2)  # [N, R, 2]

        # 正交化
        U = self.orthogonal_layer(U)  # [M, R, 2]
        V = self.orthogonal_layer(V)  # [N, R, 2]

        return U, singular_values, V


def compute_loss(U, S, V, H_ideal, lambda_ortho=1.0):
    """计算损失函数"""
    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]  # [M, R]
    V_complex = V[..., 0] + 1j * V[..., 1]  # [N, R]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]  # [M, N]

    # 重构矩阵 - 使用爱因斯坦求和避免维度错误
    H_recon = torch.einsum('mr,r,nr->mn', U_complex, S, V_complex.conj())

    # 重构误差
    recon_loss = torch.mean(torch.abs(H_ideal_complex - H_recon) ** 2)

    # 正交性约束
    R = S.shape[0]
    I = torch.eye(R, device=U.device, dtype=torch.complex64)
    U_ortho_loss = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_loss = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    total_loss = recon_loss + lambda_ortho * (U_ortho_loss + V_ortho_loss)

    return total_loss, recon_loss, U_ortho_loss, V_ortho_loss


def compute_ae_metric(U, S, V, H_ideal):
    """计算AE指标"""
    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # 重构矩阵
    H_recon = torch.einsum('mr,r,nr->mn', U_complex, S, V_complex.conj())

    # 重构误差
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro') / torch.norm(H_ideal_complex, p='fro')

    # 正交性误差
    I = torch.eye(S.shape[0], device=U.device, dtype=torch.complex64)
    U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    ae = recon_error + U_ortho_error + V_ortho_error
    return ae.item()