import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GramSchmidtLayer(nn.Module):
    """Gram-Schmidt正交化 - 不使用任何禁止算子"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [M, R, 2]
        M, R, _ = x.shape
        x_complex = x[..., 0] + 1j * x[..., 1]

        # Gram-Schmidt正交化
        Q = torch.zeros_like(x_complex)

        for j in range(R):
            v = x_complex[:, j]

            # 正交化
            for i in range(j):
                proj_coeff = torch.sum(torch.conj(Q[:, i]) * v)
                v = v - proj_coeff * Q[:, i]

            # 归一化
            norm = torch.sqrt(torch.sum(torch.abs(v) ** 2) + 1e-8)
            Q = Q.clone()  # 避免原地操作
            Q[:, j] = v / norm

        # 转换回实虚部
        return torch.stack([Q.real, Q.imag], dim=-1)


class SVDNet(nn.Module):
    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # 卷积特征提取 - 提取空间特征
        self.conv_net = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 全连接层
        self.fc_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU()
        )

        # 奇异值预测
        self.s_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, R),
            nn.Softplus()
        )

        # U矩阵预测
        self.u_net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, M * R * 2)
        )

        # V矩阵预测
        self.v_net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, N * R * 2)
        )

        # 正交化层
        self.orthogonal = GramSchmidtLayer()

    def forward(self, x):
        # x: [M, N, 2]
        # 转换为卷积输入格式
        x_conv = x.permute(2, 0, 1).unsqueeze(0)  # [1, 2, M, N]

        # 卷积特征提取
        conv_features = self.conv_net(x_conv).squeeze(0)  # [128]

        # 全连接特征
        features = self.fc_net(conv_features)  # [512]

        # 预测奇异值并排序
        s_values = self.s_net(features)  # [R]
        s_values = torch.sort(s_values, descending=True)[0]

        # 预测U和V矩阵
        u_flat = self.u_net(features)
        v_flat = self.v_net(features)

        u_matrix = u_flat.view(self.M, self.R, 2)
        v_matrix = v_flat.view(self.N, self.R, 2)

        # 正交化
        u_matrix = self.orthogonal(u_matrix)
        v_matrix = self.orthogonal(v_matrix)

        return u_matrix, s_values, v_matrix


def compute_loss(U, S, V, H_ideal, lambda_ortho=0.5):
    """计算损失函数"""
    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # SVD重构
    S_diag = torch.diag(S.to(torch.complex64))
    H_recon = U_complex @ S_diag @ V_complex.conj().T

    # 重构损失 - 使用相对误差
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro')
    ideal_norm = torch.norm(H_ideal_complex, p='fro')
    recon_loss = recon_error / (ideal_norm + 1e-8)

    # 正交性约束
    R = S.shape[0]
    I = torch.eye(R, device=U.device, dtype=torch.complex64)

    U_gram = U_complex.conj().T @ U_complex
    V_gram = V_complex.conj().T @ V_complex

    U_ortho_loss = torch.norm(U_gram - I, p='fro')
    V_ortho_loss = torch.norm(V_gram - I, p='fro')

    # 奇异值单调性约束
    if R > 1:
        monotonic_loss = torch.sum(F.relu(S[1:] - S[:-1] + 1e-6))
    else:
        monotonic_loss = torch.tensor(0.0, device=S.device)

    total_loss = (recon_loss +
                  lambda_ortho * (U_ortho_loss + V_ortho_loss) +
                  0.01 * monotonic_loss)

    return total_loss, recon_loss, U_ortho_loss, V_ortho_loss


def compute_ae_metric(U, S, V, H_ideal):
    """计算AE指标"""
    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # 重构
    S_diag = torch.diag(S.to(torch.complex64))
    H_recon = U_complex @ S_diag @ V_complex.conj().T

    # AE计算
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro') / torch.norm(H_ideal_complex, p='fro')

    I = torch.eye(S.shape[0], device=U.device, dtype=torch.complex64)
    U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    ae = recon_error + U_ortho_error + V_ortho_error
    return ae.item()