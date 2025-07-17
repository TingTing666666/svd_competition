import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OrthogonalLayer(nn.Module):
    """正交化层 - 使用QR分解"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [batch, M, R, 2] 或 [M, R, 2]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, M, R, _ = x.shape

        # 转换为复数形式
        x_complex = x[..., 0] + 1j * x[..., 1]  # [batch, M, R]

        # 对每个batch进行QR分解
        output = torch.zeros_like(x_complex)

        for b in range(batch_size):
            # QR分解得到正交矩阵Q
            Q, _ = torch.linalg.qr(x_complex[b])  # Q: [M, R]
            output[b] = Q

        # 转换回实虚部格式
        result = torch.stack([output.real, output.imag], dim=-1)

        if squeeze_output:
            result = result.squeeze(0)

        return result


class SVDNet(nn.Module):
    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # 简化的特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(M * N * 2, 512),  # 直接展平输入
            nn.ReLU(),
            nn.Linear(512, 256),
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
            nn.Softplus()  # 确保奇异值为正
        )

        # 左奇异矩阵预测网络
        self.left_singular_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, M * R * 2),  # 输出实虚部
        )

        # 右奇异矩阵预测网络
        self.right_singular_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, N * R * 2),  # 输出实虚部
        )

        # 正交化层
        self.orthogonal_layer = OrthogonalLayer()

    def forward(self, x):
        # x: [M, N, 2] 单个样本

        # 展平输入
        x_flat = x.view(-1)  # [M*N*2]

        # 特征提取
        features = self.feature_extractor(x_flat)  # [128]

        # 预测奇异值
        singular_values = self.singular_value_net(features)  # [R]

        # 确保奇异值降序排列
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
    """
    计算损失函数

    Args:
        U: [M, R, 2] 左奇异矩阵
        S: [R] 奇异值
        V: [N, R, 2] 右奇异矩阵
        H_ideal: [M, N, 2] 理想信道矩阵
        lambda_ortho: 正交性约束权重
    """
    M, R, _ = U.shape
    N, R, _ = V.shape

    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]  # [M, R]
    V_complex = V[..., 0] + 1j * V[..., 1]  # [N, R]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]  # [M, N]

    # 重构矩阵: H_recon = U @ diag(S) @ V^H
    S_diag = torch.diag(S.to(torch.complex64))  # [R, R] 转换为复数
    H_recon = U_complex @ S_diag @ V_complex.conj().T  # [M, N]

    # 重构误差
    recon_loss = torch.mean(torch.abs(H_ideal_complex - H_recon) ** 2)

    # 正交性约束
    U_ortho_loss = torch.mean(torch.abs(U_complex.conj().T @ U_complex - torch.eye(R, device=U.device)) ** 2)
    V_ortho_loss = torch.mean(torch.abs(V_complex.conj().T @ V_complex - torch.eye(R, device=V.device)) ** 2)

    total_loss = recon_loss + lambda_ortho * (U_ortho_loss + V_ortho_loss)

    return total_loss, recon_loss, U_ortho_loss, V_ortho_loss


def compute_ae_metric(U, S, V, H_ideal):
    """
    计算AE指标 (和比赛评估一致)
    """
    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # 重构矩阵
    S_diag = torch.diag(S.to(torch.complex64))
    H_recon = U_complex @ S_diag @ V_complex.conj().T

    # 重构误差
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro') / torch.norm(H_ideal_complex, p='fro')

    # 正交性误差
    I = torch.eye(S.shape[0], device=U.device)
    U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    ae = recon_error + U_ortho_error + V_ortho_error

    return ae.item()