import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OrthogonalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [M, R, 2] 避免所有原地操作
        M, R, _ = x.shape
        x_complex = x[..., 0] + 1j * x[..., 1]

        # 初始化输出矩阵
        Q = torch.zeros_like(x_complex)

        # Gram-Schmidt正交化，严格避免原地操作
        for j in range(R):
            v = x_complex[:, j]

            for i in range(j):
                q_i = Q[:, i]
                proj_coeff = torch.sum(torch.conj(q_i) * v)
                v = v - proj_coeff * q_i

            norm = torch.sqrt(torch.sum(torch.abs(v) ** 2) + 1e-8)
            Q = Q.clone()  # 创建新的张量
            Q[:, j] = v / norm

        # 转换回实虚部格式
        result = torch.stack([Q.real, Q.imag], dim=-1)
        return result


class SVDNet(nn.Module):
    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
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

        # 左奇异矩阵预测
        self.u_net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, M * R * 2)
        )

        # 右奇异矩阵预测
        self.v_net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, N * R * 2)
        )

        # 正交化层
        self.orthogonal = OrthogonalLayer()

    def forward(self, x):
        # x: [M, N, 2]
        batch_input = x.permute(2, 0, 1).unsqueeze(0)  # [1, 2, M, N]

        # 特征提取
        features = self.feature_net(batch_input).squeeze(0)  # [512]

        # 预测奇异值
        s_raw = self.s_net(features)  # [R]
        s_values, _ = torch.sort(s_raw, descending=True)

        # 预测奇异矩阵
        u_flat = self.u_net(features)
        u_matrix = u_flat.view(self.M, self.R, 2)

        v_flat = self.v_net(features)
        v_matrix = v_flat.view(self.N, self.R, 2)

        # 正交化
        u_matrix = self.orthogonal(u_matrix)
        v_matrix = self.orthogonal(v_matrix)

        return u_matrix, s_values, v_matrix


def compute_loss(U, S, V, H_ideal, lambda_ortho=0.5):
    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # SVD重构
    S_diag = torch.diag(S.to(torch.complex64))
    H_recon = U_complex @ S_diag @ V_complex.conj().T

    # 重构损失
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro')
    ideal_norm = torch.norm(H_ideal_complex, p='fro')
    recon_loss = recon_error / (ideal_norm + 1e-8)

    # 正交性损失
    R = S.shape[0]
    I = torch.eye(R, device=U.device, dtype=torch.complex64)

    U_gram = U_complex.conj().T @ U_complex
    V_gram = V_complex.conj().T @ V_complex

    U_ortho_loss = torch.norm(U_gram - I, p='fro')
    V_ortho_loss = torch.norm(V_gram - I, p='fro')

    total_loss = recon_loss + lambda_ortho * (U_ortho_loss + V_ortho_loss)

    return total_loss, recon_loss, U_ortho_loss, V_ortho_loss


def compute_ae_metric(U, S, V, H_ideal):
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