import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ComplexLinear(nn.Module):
    """复数线性层"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: [..., features, 2] 最后一维是实虚部
        real, imag = x[..., 0], x[..., 1]

        # 复数乘法：(a+bi)(c+di) = (ac-bd) + (ad+bc)i
        out_real = self.fc_r(real) - self.fc_i(imag)
        out_imag = self.fc_r(imag) + self.fc_i(real)

        return torch.stack([out_real, out_imag], dim=-1)


class ComplexConv2d(nn.Module):
    """复数卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # x: [batch, channels, H, W, 2] 最后一维是实虚部
        real, imag = x[..., 0], x[..., 1]

        # 复数卷积
        out_real = self.conv_r(real) - self.conv_i(imag)
        out_imag = self.conv_r(imag) + self.conv_i(real)

        return torch.stack([out_real, out_imag], dim=-1)


class ComplexBatchNorm2d(nn.Module):
    """复数批归一化"""

    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, x):
        # x: [batch, channels, H, W, 2]
        real, imag = x[..., 0], x[..., 1]

        real_norm = self.bn_r(real)
        imag_norm = self.bn_i(imag)

        return torch.stack([real_norm, imag_norm], dim=-1)


class SpatialAttention(nn.Module):
    """空间注意力机制 - 修复版"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, channels, H, W, 2]
        # 计算幅度
        magnitude = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-8)  # [batch, channels, H, W]

        # 在通道维度上计算统计
        avg_pool = torch.mean(magnitude, dim=1, keepdim=True)  # [batch, 1, H, W]
        max_pool = torch.max(magnitude, dim=1, keepdim=True)[0]  # [batch, 1, H, W]

        # 拼接并卷积
        concat = torch.cat([avg_pool, max_pool], dim=1)  # [batch, 2, H, W]
        attention = self.sigmoid(self.conv(concat))  # [batch, 1, H, W]

        # 扩展维度并应用注意力
        attention = attention.unsqueeze(-1)  # [batch, 1, H, W, 1]

        return x * attention


class ResidualComplexBlock(nn.Module):
    """复数残差块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels, 3, padding=1)
        self.bn1 = ComplexBatchNorm2d(channels)
        self.conv2 = ComplexConv2d(channels, channels, 3, padding=1)
        self.bn2 = ComplexBatchNorm2d(channels)
        self.attention = SpatialAttention()

    def complex_relu(self, x):
        # 复数ReLU：分别对实虚部应用ReLU
        real, imag = x[..., 0], x[..., 1]
        return torch.stack([F.relu(real), F.relu(imag)], dim=-1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.complex_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 注意力
        out = self.attention(out)

        # 残差连接
        out = out + identity
        out = self.complex_relu(out)

        return out


class GramSchmidtLayer(nn.Module):
    """修复版Gram-Schmidt正交化 - 避免inplace操作"""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: [M, R, 2] 或 [batch, M, R, 2]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, M, R, _ = x.shape

        # 转换为复数
        x_complex = x[..., 0] + 1j * x[..., 1]  # [batch, M, R]

        # 使用QR分解进行正交化（更稳定且避免inplace操作）
        output = torch.zeros_like(x_complex)

        for b in range(batch_size):
            vectors = x_complex[b]  # [M, R]

            # 使用QR分解
            try:
                Q, _ = torch.linalg.qr(vectors)  # Q是正交矩阵
                output[b] = Q
            except:
                # 如果QR分解失败，使用修改后的Gram-Schmidt
                ortho_vectors = torch.zeros_like(vectors)

                for i in range(R):
                    # 当前向量（创建副本避免inplace）
                    v = vectors[:, i].clone()  # 关键：使用clone()

                    # 减去之前所有正交向量的投影
                    for j in range(i):
                        proj_coeff = torch.sum(torch.conj(ortho_vectors[:, j]) * v)
                        v = v - proj_coeff * ortho_vectors[:, j]  # 现在这不是inplace了

                    # 归一化
                    norm = torch.sqrt(torch.sum(torch.abs(v) ** 2) + self.eps)
                    ortho_vectors[:, i] = v / norm

                output[b] = ortho_vectors

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

        # 特征提取网络 - 复数卷积
        self.feature_extractor = nn.Sequential(
            # 输入: [1, 1, M, N, 2]
            ComplexConv2d(1, 16, kernel_size=3, padding=1),
            ComplexBatchNorm2d(16),
            # 输出: [1, 16, M, N, 2]
        )

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualComplexBlock(16),
            ResidualComplexBlock(16),
            ResidualComplexBlock(16)
        ])

        # 降维卷积
        self.downsample = nn.Sequential(
            ComplexConv2d(16, 32, kernel_size=4, stride=2, padding=1),
            ComplexBatchNorm2d(32),
            ComplexConv2d(32, 64, kernel_size=4, stride=2, padding=1),
            ComplexBatchNorm2d(64),
            ComplexConv2d(64, 128, kernel_size=4, stride=2, padding=1),
            ComplexBatchNorm2d(128),
        )

        # 计算降维后的特征图大小
        conv_output_size = 128 * (M // 8) * (N // 8) * 2  # 128 channels, /8 from 3 stride-2 convs, *2 for real/imag

        # 全局特征提取
        self.global_features = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 奇异值预测网络
        self.singular_value_net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, R),
            nn.Softplus()  # 确保为正
        )

        # 左奇异矩阵预测网络
        self.left_singular_net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, M * R * 2),
            nn.Tanh()  # 限制输出范围
        )

        # 右奇异矩阵预测网络
        self.right_singular_net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, N * R * 2),
            nn.Tanh()  # 限制输出范围
        )

        # 正交化层
        self.orthogonal_layer = GramSchmidtLayer()

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def complex_relu(self, x):
        """复数ReLU激活"""
        real, imag = x[..., 0], x[..., 1]
        return torch.stack([F.relu(real), F.relu(imag)], dim=-1)

    def forward(self, x):
        # x: [M, N, 2] 单个样本
        batch_size = 1

        # 添加batch和channel维度: [1, 1, M, N, 2]
        x = x.unsqueeze(0).unsqueeze(0)

        # 特征提取
        features = self.feature_extractor(x)  # [1, 16, M, N, 2]
        features = self.complex_relu(features)

        # 残差块
        for res_block in self.res_blocks:
            features = res_block(features)

        # 降维
        features = self.downsample(features)  # [1, 128, M/8, N/8, 2]
        features = self.complex_relu(features)

        # 展平
        features_flat = features.view(batch_size, -1)  # [1, conv_output_size]

        # 全局特征
        global_feat = self.global_features(features_flat)  # [1, 512]

        # 预测奇异值
        singular_values = self.singular_value_net(global_feat).squeeze(0)  # [R]

        # 确保奇异值降序排列
        singular_values = torch.sort(singular_values, descending=True)[0]

        # 预测左奇异矩阵
        U_flat = self.left_singular_net(global_feat).squeeze(0)  # [M*R*2]
        U = U_flat.view(self.M, self.R, 2)  # [M, R, 2]

        # 预测右奇异矩阵
        V_flat = self.right_singular_net(global_feat).squeeze(0)  # [N*R*2]
        V = V_flat.view(self.N, self.R, 2)  # [N, R, 2]

        # 正交化
        U = self.orthogonal_layer(U)  # [M, R, 2]
        V = self.orthogonal_layer(V)  # [N, R, 2]

        return U, singular_values, V


def compute_loss(U, S, V, H_ideal, lambda_ortho=0.5, lambda_singular=0.1):
    """
    改进的损失函数

    Args:
        U: [M, R, 2] 左奇异矩阵
        S: [R] 奇异值
        V: [N, R, 2] 右奇异矩阵
        H_ideal: [M, N, 2] 理想信道矩阵
        lambda_ortho: 正交性约束权重
        lambda_singular: 奇异值约束权重
    """
    M, R, _ = U.shape
    N, R, _ = V.shape

    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]  # [M, R]
    V_complex = V[..., 0] + 1j * V[..., 1]  # [N, R]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]  # [M, N]

    # 重构矩阵: H_recon = U @ diag(S) @ V^H
    S_expanded = S.unsqueeze(0).expand(M, -1)  # [M, R]
    U_scaled = U_complex * S_expanded  # 逐元素相乘
    H_recon = U_scaled @ V_complex.conj().T  # [M, N]

    # 1. 重构误差 (主要损失)
    recon_loss = F.mse_loss(H_recon.real, H_ideal_complex.real) + \
                 F.mse_loss(H_recon.imag, H_ideal_complex.imag)

    # 2. 正交性约束
    I_U = torch.eye(R, device=U.device, dtype=torch.complex64)
    I_V = torch.eye(R, device=V.device, dtype=torch.complex64)

    U_gram = U_complex.conj().T @ U_complex  # [R, R]
    V_gram = V_complex.conj().T @ V_complex  # [R, R]

    U_ortho_loss = F.mse_loss(U_gram.real, I_U.real) + F.mse_loss(U_gram.imag, I_U.imag)
    V_ortho_loss = F.mse_loss(V_gram.real, I_V.real) + F.mse_loss(V_gram.imag, I_V.imag)

    # 3. 奇异值约束 (鼓励降序排列)
    singular_order_loss = torch.sum(F.relu(S[1:] - S[:-1]))  # 如果不是降序则惩罚

    # 4. 奇异值合理性约束
    H_frobenius = torch.norm(H_ideal_complex, p='fro')
    S_sum = torch.sum(S)
    singular_magnitude_loss = F.mse_loss(S_sum, H_frobenius)

    # 总损失
    total_loss = recon_loss + \
                 lambda_ortho * (U_ortho_loss + V_ortho_loss) + \
                 lambda_singular * (singular_order_loss + singular_magnitude_loss)

    return total_loss, recon_loss, U_ortho_loss, V_ortho_loss, singular_order_loss


def compute_ae_metric(U, S, V, H_ideal):
    """
    计算AE指标 (和比赛评估一致)
    """
    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # 重构矩阵
    S_expanded = S.unsqueeze(0).expand(U_complex.shape[0], -1)
    U_scaled = U_complex * S_expanded
    H_recon = U_scaled @ V_complex.conj().T

    # 重构误差
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro') / \
                  torch.norm(H_ideal_complex, p='fro')

    # 正交性误差
    I = torch.eye(S.shape[0], device=U.device, dtype=torch.complex64)
    U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    ae = recon_error + U_ortho_error + V_ortho_error

    return ae.item()