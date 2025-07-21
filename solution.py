import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OrthogonalLayer(nn.Module):
    """符合规则的正交化层 - 使用迭代正交化（修复数据类型问题）"""

    def __init__(self):
        super().__init__()
        self.num_iterations = 5  # 迭代次数

    def forward(self, x):
        # x: [M, R, 2] 或 [B, M, R, 2]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, M, R, _ = x.shape

        # 转换为复数
        x_complex = x[..., 0] + 1j * x[..., 1]

        # 使用迭代方法进行正交化
        Q_list = []
        for b in range(batch_size):
            Q = x_complex[b]

            # 迭代正交化过程（类似于Löwdin正交化）
            for _ in range(self.num_iterations):
                # 计算当前的Gram矩阵
                gram = Q.conj().T @ Q

                # 计算Gram矩阵的逆平方根（使用特征分解）
                # 这避免了直接使用SVD
                eigenvalues, eigenvectors = torch.linalg.eigh(gram)
                eigenvalues = torch.clamp(eigenvalues.real, min=1e-8)

                # ✅ 修复：确保数据类型一致
                # 构造逆平方根（保持复数类型）
                inv_sqrt_diag = torch.diag(1.0 / torch.sqrt(eigenvalues)).to(eigenvectors.dtype)
                gram_inv_sqrt = eigenvectors @ inv_sqrt_diag @ eigenvectors.conj().T

                # 更新Q
                Q = Q @ gram_inv_sqrt

            Q_list.append(Q)

        output = torch.stack(Q_list, dim=0)

        # 转换回实虚部格式
        result = torch.stack([output.real, output.imag], dim=-1)

        if squeeze_output:
            result = result.squeeze(0)

        return result


class ImprovedOrthogonalLayer(nn.Module):
    """备选方案：使用可学习的正交参数化"""

    def __init__(self, M, R):
        super().__init__()
        # 使用Cayley变换参数化
        self.W = nn.Parameter(torch.randn(M, R, 2) * 0.01)

    def forward(self, x):
        # x: [M, R, 2]
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = x.shape[0]

        # 构建反对称矩阵
        W_complex = self.W[..., 0] + 1j * self.W[..., 1]
        A = W_complex - W_complex.conj().T

        # Cayley变换: Q = (I - A)(I + A)^{-1}
        I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        Q = torch.linalg.solve(I + A, I - A)

        # 应用到输入
        x_complex = x[..., 0] + 1j * x[..., 1]
        output_list = []

        for b in range(batch_size):
            # 将输入投影到正交空间
            out = Q @ x_complex[b]
            output_list.append(out)

        output = torch.stack(output_list, dim=0)
        result = torch.stack([output.real, output.imag], dim=-1)

        if squeeze_output:
            result = result.squeeze(0)

        return result


class ChannelAwareBlock(nn.Module):
    """信道感知的特征提取块"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        # 卷积块
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 注意力机制
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // 4)
        self.fc2 = nn.Linear(out_channels // 4, out_channels)

        # 残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # 残差
        residual = self.residual(x)

        # 主分支
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 注意力分支
        b, c, h, w = out.shape
        att = self.global_pool(out).view(b, c)
        att = F.relu(self.fc1(att))
        att = torch.sigmoid(self.fc2(att)).view(b, c, 1, 1)

        # 应用注意力
        out = out * att

        # 残差连接
        out = F.relu(out + residual)

        return out


class SVDNet(nn.Module):
    """改进的SVD神经网络"""

    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # 特征提取网络
        self.feature_layers = nn.ModuleList([
            ChannelAwareBlock(2, 32, 3),
            ChannelAwareBlock(32, 64, 3),
            ChannelAwareBlock(64, 128, 3),
            ChannelAwareBlock(128, 256, 3),
        ])

        # 全局特征维度
        feature_dim = 256

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)

        # 奇异值预测网络（改进：更深的网络）
        self.singular_value_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, R),
            nn.Softplus()  # 确保奇异值为正
        )

        # 左奇异向量预测网络
        self.left_vector_net = nn.Sequential(
            nn.Linear(feature_dim + R, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, M * R * 2)
        )

        # 右奇异向量预测网络
        self.right_vector_net = nn.Sequential(
            nn.Linear(feature_dim + R, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, N * R * 2)
        )

        # 正交化层（使用QR分解版本）
        self.orthogonal_layer = OrthogonalLayer()

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        """提取特征"""
        # x: [M, N, 2] -> [1, 2, M, N]
        x = x.permute(2, 0, 1).unsqueeze(0)

        # 通过特征提取层
        for layer in self.feature_layers:
            x = layer(x)

        # 全局平均池化
        features = F.adaptive_avg_pool2d(x, 1).squeeze()  # [feature_dim]

        return features

    def forward(self, x):
        """前向传播"""
        # 提取特征
        features = self.extract_features(x)  # [feature_dim]

        # 使用注意力机制增强特征
        features_seq = features.unsqueeze(0).unsqueeze(0)  # [1, 1, feature_dim]
        enhanced_features, _ = self.multihead_attn(features_seq, features_seq, features_seq)
        enhanced_features = enhanced_features.squeeze()  # [feature_dim]

        # 预测奇异值
        singular_values = self.singular_value_net(enhanced_features)  # [R]

        # 确保奇异值降序排列
        singular_values, _ = torch.sort(singular_values, descending=True)

        # 奇异值正则化（避免过大或过小）
        singular_values = torch.clamp(singular_values, min=1e-6, max=1e3)

        # 将奇异值信息融入特征
        enhanced_features_with_s = torch.cat([enhanced_features, singular_values], dim=0)

        # 预测左奇异向量
        U_flat = self.left_vector_net(enhanced_features_with_s)  # [M*R*2]
        U = U_flat.view(self.M, self.R, 2)  # [M, R, 2]

        # 预测右奇异向量
        V_flat = self.right_vector_net(enhanced_features_with_s)  # [N*R*2]
        V = V_flat.view(self.N, self.R, 2)  # [N, R, 2]

        # 正交化处理
        U = self.orthogonal_layer(U)  # [M, R, 2]
        V = self.orthogonal_layer(V)  # [N, R, 2]

        return U, singular_values, V


def compute_loss(U, S, V, H_ideal, lambda_ortho=0.5, lambda_recon=1.0):
    """改进的损失函数"""
    M, R, _ = U.shape
    N, R, _ = V.shape

    # 转换为复数
    U_complex = U[..., 0] + 1j * U[..., 1]  # [M, R]
    V_complex = V[..., 0] + 1j * V[..., 1]  # [N, R]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]  # [M, N]

    # 重构矩阵
    S_diag = torch.diag(S.to(torch.complex64))  # [R, R]
    H_recon = U_complex @ S_diag @ V_complex.conj().T  # [M, N]

    # 1. 重构损失（使用相对误差）
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro')
    ideal_norm = torch.norm(H_ideal_complex, p='fro')
    recon_loss = recon_error / (ideal_norm + 1e-8)

    # 2. 正交性约束（软约束）
    I = torch.eye(R, device=U.device, dtype=torch.complex64)
    U_ortho_loss = torch.norm(U_complex.conj().T @ U_complex - I, p='fro') / R
    V_ortho_loss = torch.norm(V_complex.conj().T @ V_complex - I, p='fro') / R

    # 3. 奇异值单调性约束
    if R > 1:
        monotonic_loss = torch.sum(F.relu(S[1:] - S[:-1] + 1e-6))
    else:
        monotonic_loss = torch.tensor(0.0, device=S.device)

    # 4. 奇异值范围约束
    sv_penalty = torch.mean(F.relu(S - 100)) + torch.mean(F.relu(1e-6 - S))

    # 总损失
    total_loss = (lambda_recon * recon_loss +
                  lambda_ortho * (U_ortho_loss + V_ortho_loss) +
                  0.01 * monotonic_loss +
                  0.001 * sv_penalty)

    return total_loss, recon_loss, U_ortho_loss, V_ortho_loss


def compute_ae_metric(U, S, V, H_ideal):
    """计算AE指标"""
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
    I = torch.eye(S.shape[0], device=U.device, dtype=torch.complex64)
    U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    ae = recon_error + U_ortho_error + V_ortho_error

    return ae.item()