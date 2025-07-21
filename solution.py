import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------- #
# 1. 辅助模块：DropPath 和 OrthogonalLayer
# ---------------------------------------------------------------------------- #

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """按样本随机丢弃主路径（Stochastic Depth）。"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class OrthogonalLayer(nn.Module):
    """
    符合竞赛规则的正交化层。
    使用基于特征值分解的迭代方法，将输入矩阵正交化。
    这避免了直接使用被禁用的SVD或QR分解。
    """

    def __init__(self, num_iterations=10):
        super().__init__()
        self.num_iterations = num_iterations

    def forward(self, x):
        # 输入 x 的形状: [M, R, 2] 或 [B, M, R, 2]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # 转换为复数张量
        x_complex = torch.complex(x[..., 0], x[..., 1])

        # 迭代正交化 (类似Löwdin正交化)
        # 这个过程是可微分的
        Q = x_complex
        for _ in range(self.num_iterations):
            # 计算 Gram 矩阵 Q^H * Q
            gram = Q.conj().transpose(-2, -1) @ Q

            # 使用特征值分解计算 Gram 矩阵的逆平方根
            # eigh 适用于厄米矩阵，且数值稳定
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)

            # 钳制特征值以避免数值问题
            eigenvalues = torch.clamp(eigenvalues.real, min=1e-8)

            # 计算逆平方根对角阵
            inv_sqrt_diag = torch.diag_embed(1.0 / torch.sqrt(eigenvalues)).to(eigenvectors.dtype)

            # 计算 Gram 矩阵的逆平方根
            gram_inv_sqrt = eigenvectors @ inv_sqrt_diag @ eigenvectors.conj().transpose(-2, -1)

            # 更新 Q
            Q = Q @ gram_inv_sqrt

        # 转换回实虚部格式
        result = torch.stack([Q.real, Q.imag], dim=-1)

        if squeeze_output:
            result = result.squeeze(0)

        return result


# ---------------------------------------------------------------------------- #
# 2. 核心网络模块
# ---------------------------------------------------------------------------- #

class ChannelAwareBlock(nn.Module):
    """
    信道感知特征提取块 (集成了 CNN, Attention, Residual, DropPath)。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, drop_path_prob=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE-like Attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // 4)
        self.fc2 = nn.Linear(out_channels // 4, out_channels)

        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.residual(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # Attention branch
        b, c, _, _ = x.shape
        att = self.global_pool(x).view(b, c)
        att = F.relu(self.fc1(att))
        att = torch.sigmoid(self.fc2(att)).view(b, c, 1, 1)
        x = x * att

        x = self.drop_path(x)
        x = F.gelu(x + shortcut)
        return x


class SVDNet(nn.Module):
    """
    最终版 SVD 神经网络。
    """

    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # 特征提取网络 (CNN部分)
        # 线性增加DropPath概率，网络越深，丢弃概率越大
        dpr = [x.item() for x in torch.linspace(0, 0.2, 4)]
        self.feature_layers = nn.ModuleList([
            ChannelAwareBlock(2, 32, 3, drop_path_prob=dpr[0]),  # 64x64
            nn.MaxPool2d(2),  # 32x32
            ChannelAwareBlock(32, 64, 3, drop_path_prob=dpr[1]),
            nn.MaxPool2d(2),  # 16x16
            ChannelAwareBlock(64, 128, 3, drop_path_prob=dpr[2]),
            nn.MaxPool2d(2),  # 8x8
            ChannelAwareBlock(128, 256, 3, drop_path_prob=dpr[3]),
        ])

        # 全局特征维度
        feature_dim = 256

        # 多头注意力，用于融合全局特征
        self.multihead_attn = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True, dropout=0.1)

        # 奇异值预测头
        self.singular_value_net = nn.Sequential(
            nn.Linear(feature_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 128), nn.GELU(),
            nn.Linear(128, R), nn.Softplus()  # 保证奇异值为正
        )

        # 左右奇异向量预测头
        # 输入融合了特征和预测出的奇异值
        self.left_vector_net = nn.Sequential(
            nn.Linear(feature_dim + R, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, M * R * 2)
        )
        self.right_vector_net = nn.Sequential(
            nn.Linear(feature_dim + R, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, N * R * 2)
        )

        # 正交化层
        self.orthogonal_layer = OrthogonalLayer()

        # 为自动损失加权机制定义的可学习参数
        self.log_var_recon = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.log_var_ortho = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x 形状: [M, N, 2]
        # 1. 特征提取
        x = x.permute(2, 0, 1).unsqueeze(0)  # -> [1, 2, M, N]
        for layer in self.feature_layers:
            x = layer(x)

        # 2. 全局特征融合
        # AdaptiveAvgPool + MultiheadAttention
        features = F.adaptive_avg_pool2d(x, (1, 1)).squeeze()  # -> [feature_dim]
        features_seq = features.unsqueeze(0).unsqueeze(0)  # -> [1, 1, feature_dim]
        enhanced_features, _ = self.multihead_attn(features_seq, features_seq, features_seq)
        enhanced_features = enhanced_features.squeeze()  # -> [feature_dim]

        # 3. 预测奇异值
        singular_values = self.singular_value_net(enhanced_features)

        # 4. 后处理：确保奇异值降序排列并限制范围
        singular_values, _ = torch.sort(singular_values, descending=True)
        singular_values = torch.clamp(singular_values, min=1e-7)

        # 5. 预测奇异向量
        features_with_s = torch.cat([enhanced_features, singular_values], dim=-1)
        U_flat = self.left_vector_net(features_with_s)
        V_flat = self.right_vector_net(features_with_s)
        U = U_flat.view(self.M, self.R, 2)
        V = V_flat.view(self.N, self.R, 2)

        # 6. 正交化
        U_ortho = self.orthogonal_layer(U)
        V_ortho = self.orthogonal_layer(V)

        return U_ortho, singular_values, V_ortho


# ---------------------------------------------------------------------------- #
# 3. 损失函数与评估指标
# ---------------------------------------------------------------------------- #

def compute_loss(model, U_out, S_out, V_out, H_ideal):
    """
    计算总损失，集成了自动加权机制。
    """
    # 将输出和标签转换为复数
    U_complex = torch.complex(U_out[..., 0], U_out[..., 1])
    V_complex = torch.complex(V_out[..., 0], V_out[..., 1])
    H_ideal_complex = torch.complex(H_ideal[..., 0], H_ideal[..., 1])

    # --- 任务1: 重构损失 ---
    S_diag = torch.diag(S_out).to(torch.complex64)
    H_recon = U_complex @ S_diag @ V_complex.conj().T
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro')
    ideal_norm = torch.norm(H_ideal_complex, p='fro')
    recon_loss = recon_error / (ideal_norm + 1e-8)

    # --- 任务2: 正交性损失 ---
    R = U_out.shape[1]
    I = torch.eye(R, device=U_out.device, dtype=torch.complex64)
    U_ortho_loss = torch.norm(U_complex.conj().T @ U_complex - I, p='fro') / R
    V_ortho_loss = torch.norm(V_complex.conj().T @ V_complex - I, p='fro') / R
    ortho_loss = U_ortho_loss + V_ortho_loss

    # --- 自动加权 ---
    # 根据 "Multi-Task Learning Using Uncertainty to Weigh Losses..."
    # L(W, sigma) = L_i(W) / (2*sigma_i^2) + log(sigma_i)
    # 这里我们简化为 L = exp(-log_var)*L_task + log_var
    loss1 = torch.exp(-model.log_var_recon) * recon_loss + model.log_var_recon
    loss2 = torch.exp(-model.log_var_ortho) * ortho_loss + model.log_var_ortho

    total_loss = loss1 + loss2

    return total_loss, recon_loss, ortho_loss


def compute_ae_metric(U, S, V, H_ideal):
    """计算官方的 AE 指标，用于验证。"""
    # 转换为复数
    U_complex = torch.complex(U[..., 0], U[..., 1])
    V_complex = torch.complex(V[..., 0], V[..., 1])
    H_ideal_complex = torch.complex(H_ideal[..., 0], H_ideal[..., 1])

    # 重构矩阵
    S_diag = torch.diag(S).to(torch.complex64)
    H_recon = U_complex @ S_diag @ V_complex.conj().T

    # 重构误差
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro') / torch.norm(H_ideal_complex, p='fro')

    # 正交性误差
    I = torch.eye(S.shape[0], device=U.device, dtype=torch.complex64)
    U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    ae = recon_error + U_ortho_error + V_ortho_error
    return ae.item()