import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GramSchmidt(nn.Module):
    def __init__(self, iterations=3):
        super().__init__()
        self.iterations = iterations

    def forward(self, x):
        # x: [M, R, 2]
        M, R, _ = x.shape
        x_complex = x[..., 0] + 1j * x[..., 1]

        # Modified Gram-Schmidt without inplace operations
        Q = torch.zeros_like(x_complex)
        for i in range(R):
            q = x_complex[:, i].clone()
            for j in range(i):
                proj_coeff = torch.sum(torch.conj(Q[:, j]) * q)
                q = q - proj_coeff * Q[:, j]
            norm = torch.sqrt(torch.sum(torch.abs(q) ** 2) + 1e-10)
            Q = Q.clone()  # Avoid inplace modification
            Q[:, i] = q / norm

        # Iterative refinement without inplace operations
        for _ in range(self.iterations):
            gram = Q.conj().T @ Q
            I = torch.eye(R, device=Q.device, dtype=Q.dtype)
            correction_matrix = (gram - I) * 0.4
            correction = Q @ correction_matrix
            Q = Q - correction  # Create new tensor

            # Re-normalize without inplace
            norms = torch.sqrt(torch.sum(torch.abs(Q) ** 2, dim=0, keepdim=True) + 1e-10)
            Q = Q / norms

        return torch.stack([Q.real, Q.imag], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + residual)


class SVDNet(nn.Module):
    def __init__(self, M=64, N=64, R=32):
        super().__init__()
        self.M = M
        self.N = N
        self.R = R

        # Multi-scale feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            ResBlock(64, 128),
            ResBlock(128, 256),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Enhanced feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.ReLU()
        )

        # Singular value prediction
        self.s_net = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, R),
            nn.Softplus()
        )

        # U matrix prediction with context
        self.u_net = nn.Sequential(
            nn.Linear(768 + R, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, M * R * 2)
        )

        # V matrix prediction with context
        self.v_net = nn.Sequential(
            nn.Linear(768 + R, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, N * R * 2)
        )

        self.orthogonal = GramSchmidt(iterations=3)
        self.s_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: [M, N, 2]
        x_conv = x.permute(2, 0, 1).unsqueeze(0)  # [1, 2, M, N]

        features = self.backbone(x_conv).squeeze(0)
        features = self.feature_net(features)

        # Predict singular values
        s_raw = self.s_net(features)
        s_values = torch.sort(s_raw * torch.abs(self.s_scale), descending=True)[0]

        # Predict U and V with singular value context
        context = torch.cat([features, s_values.detach()], dim=-1)

        U_flat = self.u_net(context)
        V_flat = self.v_net(context)

        U = U_flat.view(self.M, self.R, 2)
        V = V_flat.view(self.N, self.R, 2)

        # Orthogonalize
        U = self.orthogonal(U)
        V = self.orthogonal(V)

        return U, s_values, V


def compute_loss(U, S, V, H_ideal, lambda_ortho=1.0, lambda_energy=0.1):
    # Convert to complex
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # SVD reconstruction
    H_recon = torch.einsum('mr,r,nr->mn', U_complex, S, V_complex.conj())

    # Reconstruction loss - multiple norms
    recon_error_fro = torch.norm(H_ideal_complex - H_recon, p='fro')
    recon_error_2 = torch.norm(H_ideal_complex - H_recon, p=2)
    ideal_norm = torch.norm(H_ideal_complex, p='fro')
    recon_loss = (recon_error_fro + 0.2 * recon_error_2) / (ideal_norm + 1e-8)

    # Strict orthogonality constraints
    R = S.shape[0]
    I = torch.eye(R, device=U.device, dtype=torch.complex64)

    U_gram = U_complex.conj().T @ U_complex
    V_gram = V_complex.conj().T @ V_complex

    U_ortho_loss = torch.norm(U_gram - I, p='fro') + 0.5 * torch.norm(U_gram - I, p=2)
    V_ortho_loss = torch.norm(V_gram - I, p='fro') + 0.5 * torch.norm(V_gram - I, p=2)

    # Diagonal should be 1
    U_diag_loss = torch.norm(torch.diag(U_gram) - torch.ones(R, device=U.device, dtype=torch.complex64))
    V_diag_loss = torch.norm(torch.diag(V_gram) - torch.ones(R, device=V.device, dtype=torch.complex64))

    # Singular value constraints
    monotonic_loss = torch.sum(F.relu(S[1:] - S[:-1] + 1e-6)) if R > 1 else torch.tensor(0.0, device=S.device)
    positive_loss = torch.sum(F.relu(1e-6 - S))

    # Energy consistency
    ideal_energy = torch.sum(torch.abs(H_ideal_complex) ** 2)
    recon_energy = torch.sum(S ** 2)
    energy_loss = torch.abs(ideal_energy - recon_energy) / (ideal_energy + 1e-8)

    total_loss = (recon_loss +
                  lambda_ortho * (U_ortho_loss + V_ortho_loss + 0.3 * (U_diag_loss + V_diag_loss)) +
                  0.03 * (monotonic_loss + positive_loss) +
                  lambda_energy * energy_loss)

    return total_loss, recon_loss, U_ortho_loss, V_ortho_loss


def compute_ae_metric(U, S, V, H_ideal):
    # Convert to complex
    U_complex = U[..., 0] + 1j * U[..., 1]
    V_complex = V[..., 0] + 1j * V[..., 1]
    H_ideal_complex = H_ideal[..., 0] + 1j * H_ideal[..., 1]

    # Reconstruction
    H_recon = torch.einsum('mr,r,nr->mn', U_complex, S, V_complex.conj())

    # AE metric exactly as in competition
    recon_error = torch.norm(H_ideal_complex - H_recon, p='fro') / torch.norm(H_ideal_complex, p='fro')

    I = torch.eye(S.shape[0], device=U.device, dtype=torch.complex64)
    U_ortho_error = torch.norm(U_complex.conj().T @ U_complex - I, p='fro')
    V_ortho_error = torch.norm(V_complex.conj().T @ V_complex - I, p='fro')

    ae = recon_error + U_ortho_error + V_ortho_error
    return ae.item()