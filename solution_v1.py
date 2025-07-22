import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StrictOrthogonal(nn.Module):
    """严格正交化 - 多次迭代确保数值稳定"""

    def __init__(self, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations

    def forward(self, x):
        # x: [M, R, 2]
        M, R, _ = x.shape
        x_complex = x[..., 0] + 1j * x[..., 1]

        # 初始Gram-Schmidt
        Q = torch.zeros_like(x_complex)
        for j in range(R):
            v = x_complex[:, j]
            for i in range(j):
                proj_coeff = torch.sum(torch.conj(Q[:, i]) * v)
                v = v - proj_coeff * Q[:, i]
            norm = torch.sqrt(torch.sum(torch.abs(v) ** 2) + 1e-10)
            Q = Q.clone()
            Q[:, j] = v / norm

        # 多次重正交化提升数值精度
        for _ in range(self.num_iterations):
            Q_new = torch.zeros_like(Q)
            for j in range(R):
                v = Q[:, j]
                for i in range(R):
                    if i != j:
                        proj_coeff = torch.sum(torch.conj(Q[:, i]) * v)
                        v = v - 0.5 * proj_coeff * Q[:, i]
                norm = torch.sqrt(torch.sum(torch.abs(v) ** 2) + 1e-10)
                Q_new[:, j] = v / norm
            Q = Q_new

        return torch.stack([Q.real, Q.imag], dim=-1)


class SVDNet(nn.Module):
    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # 多尺度特征提取
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # 全局特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )

        # 奇异值网络 - 加强约束
        self.s_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, R),
            nn.Softplus()
        )

        # U和V网络
        self.u_net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, M * R * 2)
        )

        self.v_net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, N * R * 2)
        )

        # 严格正交化
        self.orthogonal = StrictOrthogonal(num_iterations=3)

        # 奇异值排序约束
        self.s_temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: [M, N, 2]
        x_conv = x.permute(2, 0, 1).unsqueeze(0)  # [1, 2, M, N]

        # 特征提取
        f1 = F.relu(self.conv1(x_conv))  # [1, 32, M, N]
        f2 = F.relu(self.conv2(f1))  # [1, 64, M, N]
        f3 = F.relu(self.conv3(f2))  # [1, 128, M, N]

        # 全局特征
        global_feat = self.global_pool(f3).flatten()  # [128]
        features = self.feature_fc(global_feat)  # [512]

        # 预测奇异值并强制排序
        s_logits = self.s_net(features)  # [R]
        s_values = torch.sort(s_logits * torch.abs(self.s_temperature), descending=True)[0]

        # 预测U和V
        u_flat = self.u_net(features)
        v_flat = self.v_net(features)

        u_matrix = u_flat.view(self.M, self.R, 2)
        v_matrix = v_flat.view(self.N, self.R, 2)

        # 严格正交化
        u_matrix = self.orthogonal(u_matrix)
        v_matrix = self.orthogonal(v_matrix)

        return u_matrix, s_values, v_matrix


def compute_loss(U, S, V, H_ideal, lambda_ortho=0.8, lambda_monotonic=0.1):
    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # 重构损失
    S_diag = torch.diag(S.to(torch.complex64))
    H_recon = U_complex @ S_diag @ V_complex.conj().T

    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro')
    ideal_norm = torch.norm(H_ideal_complex, p='fro')
    recon_loss = recon_error / (ideal_norm + 1e-8)

    # 严格正交性损失
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
                  lambda_monotonic * monotonic_loss)

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