import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnhancedOrthogonal(nn.Module):
    """增强正交化 - 平衡速度和精度"""

    def __init__(self, num_iterations=2):
        super().__init__()
        self.num_iterations = num_iterations

    def forward(self, x):
        # x: [M, R, 2]
        M, R, _ = x.shape
        x_complex = x[..., 0] + 1j * x[..., 1]

        # 初始归一化
        norms = torch.sqrt(torch.sum(torch.abs(x_complex) ** 2, dim=0, keepdim=True) + 1e-8)
        Q = x_complex / norms

        # 多次迭代改进正交性
        for _ in range(self.num_iterations):
            # 计算Gram矩阵
            gram = Q.conj().T @ Q
            # 对角化处理
            gram_diag = torch.diag(torch.diag(gram))
            # 正交化修正
            correction = Q @ (gram - gram_diag) * 0.5
            Q = Q - correction

            # 重新归一化
            norms = torch.sqrt(torch.sum(torch.abs(Q) ** 2, dim=0, keepdim=True) + 1e-8)
            Q = Q / norms

        return torch.stack([Q.real, Q.imag], dim=-1)


class SVDNet(nn.Module):
    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # 增强卷积特征提取 - 修正输入处理
        self.conv_backbone = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 多尺度特征处理
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # 奇异值预测网络
        self.s_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, R),
            nn.Softplus()
        )

        # U矩阵预测网络 - 加入奇异值信息
        self.u_net = nn.Sequential(
            nn.Linear(512 + R, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, M * R * 2)
        )

        # V矩阵预测网络 - 加入奇异值信息
        self.v_net = nn.Sequential(
            nn.Linear(512 + R, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, N * R * 2)
        )

        # 增强正交化
        self.orthogonal = EnhancedOrthogonal(num_iterations=2)

        # 可学习缩放因子
        self.s_scale = nn.Parameter(torch.ones(1))

        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        # x: [M, N, 2] - 根据文档，这是单个样本的输入格式
        # 转换为卷积输入格式 [1, 2, M, N]
        x_conv = x.permute(2, 0, 1).unsqueeze(0)  # [M, N, 2] -> [2, M, N] -> [1, 2, M, N]

        # 增强特征提取
        conv_features = self.conv_backbone(x_conv).squeeze(0)  # [256]
        features = self.feature_enhance(conv_features)  # [512]

        # 预测奇异值并强制降序
        s_values = self.s_net(features)  # [R]
        s_values = torch.sort(s_values * torch.abs(self.s_scale), descending=True)[0]

        # 特征融合 - 将奇异值信息融入U、V预测
        enhanced_features = torch.cat([features, s_values.detach()], dim=-1)  # [512+R]

        # 预测U和V矩阵
        U_flat = self.u_net(enhanced_features)
        V_flat = self.v_net(enhanced_features)

        U = U_flat.view(self.M, self.R, 2)
        V = V_flat.view(self.N, self.R, 2)

        # 增强正交化
        U = self.orthogonal(U)
        V = self.orthogonal(V)

        return U, s_values, V


def compute_loss(U, S, V, H_ideal, lambda_ortho=0.8, lambda_energy=0.05):
    """增强损失函数"""
    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]  # [M, R]
    V_complex = V[..., 0] + 1j * V[..., 1]  # [N, R]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]  # [M, N]

    # SVD重构
    H_recon = torch.einsum('mr,r,nr->mn', U_complex, S, V_complex.conj())

    # 1. 多范数重构损失
    recon_error_fro = torch.norm(H_ideal_complex - H_recon, p='fro')
    recon_error_2 = torch.norm(H_ideal_complex - H_recon, p=2)
    ideal_norm = torch.norm(H_ideal_complex, p='fro')
    recon_loss = (recon_error_fro + 0.2 * recon_error_2) / (ideal_norm + 1e-8)

    # 2. 严格正交性约束
    R = S.shape[0]
    I = torch.eye(R, device=U.device, dtype=torch.complex64)

    U_gram = U_complex.conj().T @ U_complex
    V_gram = V_complex.conj().T @ V_complex

    U_ortho_loss = torch.norm(U_gram - I, p='fro')
    V_ortho_loss = torch.norm(V_gram - I, p='fro')

    # 3. 奇异值约束
    monotonic_loss = torch.tensor(0.0, device=S.device)
    if R > 1:
        monotonic_loss = torch.sum(F.relu(S[1:] - S[:-1] + 1e-6))

    positive_loss = torch.sum(F.relu(1e-6 - S))

    # 4. 能量一致性
    ideal_energy = torch.sum(torch.abs(H_ideal_complex) ** 2)
    recon_energy = torch.sum(S ** 2)
    energy_loss = torch.abs(ideal_energy - recon_energy) / (ideal_energy + 1e-8)

    total_loss = (recon_loss +
                  lambda_ortho * (U_ortho_loss + V_ortho_loss) +
                  0.02 * (monotonic_loss + positive_loss) +
                  lambda_energy * energy_loss)

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