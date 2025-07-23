# solution.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ["SVDNet", "compute_loss", "compute_ae_metric"]

# -------------------- Hyperparameters --------------------
DEFAULT_M = 64
DEFAULT_N = 64
DEFAULT_R = 32
DEFAULT_HIDDEN = 48
ORTHO_ITERS = 1

# -------------------- Complex ↔ Real/Imag Utilities --------------------
def _to_complex(x: torch.Tensor) -> torch.Tensor:
    return x[..., 0] + 1j * x[..., 1]

def _to_ri(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=-1)

# -------------------- Symmetric Orthogonalization --------------------
@torch.no_grad()
def symmetric_orthogonalize(Q_ri: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """
    Approximate orthogonalization without QR/SVD.
    Q_ri: [B, M, R, 2] or [M, R, 2]
    """
    squeeze = (Q_ri.dim() == 3)
    if squeeze:
        Q_ri = Q_ri.unsqueeze(0)
    Q = _to_complex(Q_ri).to(torch.complex64)  # ensure full‑precision complex
    B, M, R = Q.shape
    I = torch.eye(R, device=Q.device, dtype=Q.dtype)

    for _ in range(iters):
        gram = Q.conj().transpose(-2, -1) @ Q
        Q = Q - 0.5 * (Q @ (gram - I))

    out = _to_ri(Q)
    return out.squeeze(0) if squeeze else out

# -------------------- MLP Head --------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, act_last=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim * 2)
        self.fc2 = nn.Linear(in_dim * 2, in_dim)
        self.fc3 = nn.Linear(in_dim, out_dim)
        self.act_last = act_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = self.fc3(x)
        return self.act_last(x) if self.act_last else x

# -------------------- SVDNet --------------------
class SVDNet(nn.Module):
    """
    Supports both SVDNet() and SVDNet(M, N, R).
    In training mode: standard forward (no projection or sorting).
    In eval mode: does symmetric orthogonalization + S-sort.
    """
    def __init__(self,
                 M: int = None,
                 N: int = None,
                 R: int = None,
                 hidden: int = DEFAULT_HIDDEN,
                 ortho_iter: int = ORTHO_ITERS):
        super().__init__()
        self.M = M or DEFAULT_M
        self.N = N or DEFAULT_N
        self.R = R or DEFAULT_R
        self.hidden = hidden
        self.ortho_iter = ortho_iter
        self._build_layers(self.M, self.N, self.R)

    def _build_layers(self, M: int, N: int, R: int):
        self.M, self.N, self.R = M, N, R
        self.encoder = nn.Sequential(
            nn.Conv2d(2, self.hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden, 3, padding=1, groups=self.hidden), nn.GELU(),
            nn.Conv2d(self.hidden, self.hidden, 1),   nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        in_dim = self.hidden
        self.head_S = MLPHead(in_dim, self.R, act_last=F.softplus)
        self.head_U = MLPHead(in_dim, self.M * self.R * 2)
        self.head_V = MLPHead(in_dim, self.N * self.R * 2)

        # Xavier initialization for all Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def maybe_rebuild(self, M: int, N: int):
        if M != self.M or N != self.N:
            self._build_layers(M, N, self.R)
            self.to(next(self.parameters()).device)

    def forward(self, H_ri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Accepts [M,N,2] or [B,M,N,2]
        squeeze = False
        if H_ri.dim() == 3:
            H_ri = H_ri.unsqueeze(0)
            squeeze = True

        B, M, N, _ = H_ri.shape
        self.maybe_rebuild(M, N)

        x = H_ri.permute(0, 3, 1, 2)       # [B,2,M,N]
        feat = self.encoder(x).view(B, -1) # [B,hidden]

        S = self.head_S(feat)              # [B,R]
        U = self.head_U(feat).view(B, self.M, self.R, 2)
        V = self.head_V(feat).view(B, self.N, self.R, 2)

        if not self.training:
            # Inference: orthogonalize + sort
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

# -------------------- Loss & AE Metric --------------------
def compute_loss(U: torch.Tensor,
                 S: torch.Tensor,
                 V: torch.Tensor,
                 H_label: torch.Tensor,
                 lambda_ortho: float = 0.4):
    """
    compute_loss returns:
      total_loss,
      rec_mean,
      u_pen_mean,
      v_pen_mean
    with AE = rec + u_pen + v_pen.
    All complex operations are done in full precision.
    """
    with torch.amp.autocast(device_type="cuda", enabled=False):
        def fro(x): return torch.norm(x, dim=(-2, -1))
        squeeze = (U.dim() == 3)
        if squeeze:
            U = U.unsqueeze(0)
            S = S.unsqueeze(0)
            V = V.unsqueeze(0)
            H_label = H_label.unsqueeze(0)

        Uc = _to_complex(U).to(torch.complex64)
        Vc = _to_complex(V).to(torch.complex64)
        Hc = _to_complex(H_label).to(torch.complex64)
        Sd = torch.diag_embed(S.to(torch.complex64))

        H_rec = Uc @ Sd @ Vc.conj().transpose(-2, -1)
        rec = fro(Hc - H_rec) / (fro(Hc) + 1e-12)

        R = S.shape[-1]
        I = torch.eye(R, device=S.device, dtype=Uc.dtype)
        u_pen = torch.norm(Uc.conj().transpose(-2, -1) @ Uc - I, dim=(-2, -1))
        v_pen = torch.norm(Vc.conj().transpose(-2, -1) @ Vc - I, dim=(-2, -1))

        total = rec.mean() + lambda_ortho * (u_pen.mean() + v_pen.mean())

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
