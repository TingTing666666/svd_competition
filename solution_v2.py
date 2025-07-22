import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        att = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * att


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * att


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.ca(out)
        out = self.sa(out)
        return F.relu(out + residual)


class FastOrthogonal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
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

        # 增强的特征提取网络
        self.backbone = nn.Sequential(
            ResidualBlock(2, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # 多头预测
        self.s_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, R),
            nn.Softplus()
        )

        self.u_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, M * R * 2)
        )

        self.v_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, N * R * 2)
        )

        self.orthogonal = FastOrthogonal()

    def forward(self, x):
        # x: [M, N, 2]
        x_conv = x.permute(2, 0, 1).unsqueeze(0)  # [1, 2, M, N]

        # 特征提取
        features = self.backbone(x_conv).squeeze(0)  # [256]
        features = self.feature_enhance(features)  # [512]

        # 多头预测
        s_values = self.s_head(features)  # [R]
        s_values = torch.sort(s_values, descending=True)[0]

        u_flat = self.u_head(features)
        v_flat = self.v_head(features)

        u_matrix = u_flat.view(self.M, self.R, 2)
        v_matrix = v_flat.view(self.N, self.R, 2)

        # 正交化
        u_matrix = self.orthogonal(u_matrix)
        v_matrix = self.orthogonal(v_matrix)

        return u_matrix, s_values, v_matrix


def compute_loss(U, S, V, H_ideal, lambda_ortho=0.6):
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    S_diag = torch.diag(S.to(torch.complex64))
    H_recon = U_complex @ S_diag @ V_complex.conj().T

    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro')
    ideal_norm = torch.norm(H_ideal_complex, p='fro')
    recon_loss = recon_error / (ideal_norm + 1e-8)

    R = S.shape[0]
    I = torch.eye(R, device=U.device, dtype=torch.complex64)

    U_gram = U_complex.conj().T @ U_complex
    V_gram = V_complex.conj().T @ V_complex

    U_ortho_loss = torch.norm(U_gram - I, p='fro')
    V_ortho_loss = torch.norm(V_gram - I, p='fro')

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