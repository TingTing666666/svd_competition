# solution.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ["SVDNet", "compute_loss", "compute_ae_metric"]

# --------------------
# 超参
# --------------------
DEFAULT_M = 64
DEFAULT_N = 64
DEFAULT_R = 32
# 增大隐藏通道以提升容量
DEFAULT_HIDDEN = 256
# 残差块数
NUM_RES_BLOCKS = 4
# 正交迭代次数
ORTHO_ITERS = 1


# --------------------
# 复数工具
# --------------------
def _to_complex(x: torch.Tensor) -> torch.Tensor:
    return x[..., 0] + 1j * x[..., 1]

def _to_ri(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=-1)


# --------------------
# 近似正交化（eval 时用）
# --------------------
@torch.no_grad()
def symmetric_orthogonalize(Q_ri: torch.Tensor, iters: int = ORTHO_ITERS) -> torch.Tensor:
    squeeze = (Q_ri.dim() == 3)
    if squeeze:
        Q_ri = Q_ri.unsqueeze(0)
    # 转 complex64，full‑precision 计算
    Q = _to_complex(Q_ri).to(torch.complex64)  # [B,M,R]
    B, M, R = Q.shape
    I = torch.eye(R, device=Q.device, dtype=Q.dtype)
    for _ in range(iters):
        G = Q.conj().transpose(-2, -1) @ Q        # [B,R,R]
        Q = Q - 0.5 * (Q @ (G - I))
    out = _to_ri(Q)  # back to real/imag planes
    return out.squeeze(0) if squeeze else out


# --------------------
# 深度残差卷积块
# --------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + x)


# --------------------
# 强化 MLP 头
# --------------------
class DeepMLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, act_last=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.act_last = act_last
        # 初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)
    def forward(self, x):
        x = self.net(x)
        return self.act_last(x) if self.act_last else x


# --------------------
# 主网络
# --------------------
# 在 solution.py 中，把 SVDNet 构造函数改为：
class SVDNet(nn.Module):
    def __init__(self, M=None, N=None, R=None,
                 hidden=96, num_blocks=2, ortho_iter=1):
        super().__init__()
        self.M = M or 64; self.N = N or 64; self.R = R or 32
        self.hidden = hidden
        self.ortho_iter = ortho_iter
        self.num_blocks = num_blocks
        self._build_layers()

    def _build_layers(self):
        # 深度残差编码器：2 个块
        layers = [nn.Conv2d(2, self.hidden, 3, padding=1, bias=False),
                  nn.BatchNorm2d(self.hidden), nn.GELU()]
        for _ in range(self.num_blocks):
            layers.append(ResBlock(self.hidden))
        layers += [
            nn.AdaptiveAvgPool2d(1),
        ]
        self.encoder = nn.Sequential(*layers)

        # 浅层 MLP 头
        feat_dim = self.hidden
        mid_dim = feat_dim
        self.head_S = nn.Sequential(
            nn.Linear(feat_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, self.R),
            nn.Softplus()
        )
        self.head_U = nn.Sequential(
            nn.Linear(feat_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, self.M*self.R*2),
        )
        self.head_V = nn.Sequential(
            nn.Linear(feat_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, self.N*self.R*2),
        )

    def forward(self, H_ri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # H_ri: [M,N,2] 或 [B,M,N,2]
        squeeze = False
        if H_ri.dim() == 3:
            H_ri = H_ri.unsqueeze(0)
            squeeze = True

        B, M, N, _ = H_ri.shape
        x = H_ri.permute(0, 3, 1, 2)  # [B,2,M,N]
        feat = self.encoder(x).view(B, -1)  # [B, hidden]

        S = self.head_S(feat)  # [B,R]
        U = self.head_U(feat).view(B, self.M, self.R, 2)
        V = self.head_V(feat).view(B, self.N, self.R, 2)

        if not self.training:
            # 推理：正交化 + 排序
            U = symmetric_orthogonalize(U, self.ortho_iter)
            V = symmetric_orthogonalize(V, self.ortho_iter)
            S, idx = torch.sort(S, dim=-1, descending=True)
            idx_u = idx.unsqueeze(1).unsqueeze(-1).expand(-1, U.size(1), -1, 2)
            idx_v = idx.unsqueeze(1).unsqueeze(-1).expand(-1, V.size(1), -1, 2)
            U = torch.gather(U, 2, idx_u)
            V = torch.gather(V, 2, idx_v)

        if squeeze:
            U = U.squeeze(0)
            S = S.squeeze(0)
            V = V.squeeze(0)
        return U, S, V


# --------------------
# 训练用损失 & AE 评估
# --------------------
def compute_loss(U: torch.Tensor,
                 S: torch.Tensor,
                 V: torch.Tensor,
                 H_label: torch.Tensor,
                 lambda_ortho: float = 0.1):
    """
    Loss = rec + lambda_ortho*(u_pen+v_pen)
    rec = ||H-Hrec||_F/||H||_F
    """
    # 关闭 autocast，确保 full‑precision complex
    with torch.amp.autocast(device_type="cuda", enabled=False):
        def fro(x): return torch.norm(x, dim=(-2, -1))
        squeeze = (U.dim() == 3)
        if squeeze:
            U = U.unsqueeze(0); S = S.unsqueeze(0)
            V = V.unsqueeze(0); H_label = H_label.unsqueeze(0)

        Uc = _to_complex(U).to(torch.complex64)
        Vc = _to_complex(V).to(torch.complex64)
        Hc = _to_complex(H_label).to(torch.complex64)
        Sd = torch.diag_embed(S.to(torch.complex64))

        Hrec = Uc @ Sd @ Vc.conj().transpose(-2, -1)
        rec = fro(Hc - Hrec) / (fro(Hc) + 1e-12)

        R = S.shape[-1]
        I = torch.eye(R, device=U.device, dtype=Uc.dtype)
        u_pen = torch.norm(Uc.conj().transpose(-2, -1)@Uc - I, dim=(-2, -1))
        v_pen = torch.norm(Vc.conj().transpose(-2, -1)@Vc - I, dim=(-2, -1))

        total = rec.mean() + lambda_ortho*(u_pen.mean() + v_pen.mean())
        if squeeze:
            return total, rec.mean(), u_pen.mean(), v_pen.mean()
        return total, rec.mean(), u_pen.mean(), v_pen.mean()

def compute_ae_metric(U: torch.Tensor,
                      S: torch.Tensor,
                      V: torch.Tensor,
                      H_label: torch.Tensor) -> float:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        _, rec, u_pen, v_pen = compute_loss(U, S, V, H_label, lambda_ortho=1.0)
        return (rec + u_pen + v_pen).item()
