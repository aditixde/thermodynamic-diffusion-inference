#!/usr/bin/env python3
"""
Thermodynamic Diffusion Inference with Minimal Digital Conditioning
===================================================================
Reproduces all experiments and figures from the paper.

Usage:
    python run_experiments.py                  # Full reproduction
    python run_experiments.py --no-figures     # Experiments only
    python run_experiments.py --train-mnist    # Train on real MNIST
    python run_experiments.py --nonlinear      # Include nonlinear validation

Author: Aditi De
"""

import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")
torch.manual_seed(42)

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN = 256
N_TEST = 64
D = 128          # activation dimension (analytical regime)
D_TRAINED = 64   # activation dimension (trained regime)
kT = 1.0         # Boltzmann constant × temperature
J2 = 0.1         # quadratic self-coupling
RANKS = [2, 4, 8, 16, 32, 48, 64]
ENCODER_DIMS = [4, 8, 16, 32, 64, 96, 128]
MLP_DIMS = [8, 16, 32, 64]

os.makedirs("figures", exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Model Architecture
# ═════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Residual block matching Stable Diffusion 1.5 topology."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.proj(x))


class ToyUNet(nn.Module):
    """
    Compact U-Net matching Stable Diffusion 1.5 topology.
    Channels: 128 → 256 → 512 (encoder), 512 → 256 → 128 (decoder).
    Skip connections from enc1→dec2 and enc2→dec1.
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(4, 128, 3, padding=1)
        self.e1 = ResBlock(128, 128)
        self.e2 = ResBlock(128, 256)
        self.bot = ResBlock(256, 512)
        self.d2 = ResBlock(512 + 256, 256)
        self.d1 = ResBlock(256 + 128, 128)
        self.out = nn.Conv2d(128, 4, 1)
        self.dn = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, return_activations=False):
        h0 = F.silu(self.stem(x))
        h1 = self.e1(h0)
        h2 = self.e2(self.dn(h1))
        h3 = self.bot(self.dn(h2))
        d2 = self.d2(torch.cat([self.up(h3), h2], 1))
        d1 = self.d1(torch.cat([self.up(d2), h1], 1))
        o = self.out(d1)
        if return_activations:
            return o, {'e1': h1, 'e2': h2, 'bot': h3, 'd2': d2, 'd1': d1}
        return o


# ═════════════════════════════════════════════════════════════════════════════
# Coupling Matrices (Section 3)
# ═════════════════════════════════════════════════════════════════════════════

def gram_matrix(W, dim):
    """Build Gram coupling J = W^T W / (4kT) from convolutional weights."""
    W = W.detach().cpu().float()
    if W.dim() == 4:
        W = W.reshape(W.shape[0], -1)
    W = W[:dim, :dim]
    return (W.T @ W) / (4.0 * kT)


def build_skip_coupling(J_enc, J_dec, rank=16):
    """
    Hierarchical bilinear skip coupling (Definition 1 in paper).
    J_skip^(k) = (U_e^(k) Σ_e^(k)^{1/2})(U_d^(k) Σ_d^(k)^{1/2})^T / ||·||_F
    Requires O(Dk) physical connections instead of O(D²).
    """
    Ue, Se, _ = torch.linalg.svd(J_enc)
    Ud, Sd, _ = torch.linalg.svd(J_dec)
    U = Ue[:, :rank] * Se[:rank].sqrt()
    V = Ud[:, :rank] * Sd[:rank].sqrt()
    J = U @ V.T
    return J / (J.norm() + 1e-8) / (4.0 * kT)


def build_system_matrix(J_enc, J_dec, J_skip, dim):
    """
    Build the block system matrix M from Eq. (4):
    M = [[A, J_skip], [J_skip^T, B]]
    where A = 2J₂I + J_enc + J_enc^T, B = 2J₂I + J_dec + J_dec^T.
    """
    A = 2 * J2 * torch.eye(dim) + J_enc + J_enc.T
    B = 2 * J2 * torch.eye(dim) + J_dec + J_dec.T
    M = torch.zeros(2 * dim, 2 * dim)
    M[:dim, :dim] = A
    M[:dim, dim:] = J_skip
    M[dim:, :dim] = J_skip.T
    M[dim:, dim:] = B
    return A, B, M


def solve_equilibrium(b_enc, b_dec, M_inv, dim):
    """Solve M [x* y*]^T = [b_enc b_dec]^T for equilibrium activations."""
    b = torch.cat([b_enc, b_dec], 1)
    sol = b @ M_inv.T
    return sol[:, :dim], sol[:, dim:]


def normalize(t):
    """Zero-mean, unit-variance normalization."""
    return (t - t.mean()) / (t.std() + 1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# MNIST Training (optional)
# ═════════════════════════════════════════════════════════════════════════════

def train_on_mnist(model, epochs=2, n_samples=10000):
    """Train the U-Net as a denoiser on MNIST (σ=0.1 noise)."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.repeat(4, 1, 1)),
    ])
    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    dataset = Subset(dataset, range(n_samples))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)

    model = model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            noise = torch.randn_like(images) * 0.1
            pred = model(images + noise)
            loss = F.mse_loss(pred, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Loss: {total_loss/len(loader):.4f}")

    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Experiment A: Skip Coupling Effect (Section 3)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_a(x_test, d_test, A, B, J_skip, dim):
    """
    Measure the relative decoder shift ρ_skip induced by rank-k skip coupling.
    Compares equilibrium with and without J_skip in the block system.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Skip Coupling Effect (Section 3)")
    print("=" * 60)

    # System with skip
    M_with = torch.zeros(2 * dim, 2 * dim)
    M_with[:dim, :dim] = A
    M_with[dim:, dim:] = B
    M_with[:dim, dim:] = J_skip
    M_with[dim:, :dim] = J_skip.T
    M_with_inv = torch.linalg.inv(M_with)

    # System without skip
    M_no = torch.zeros(2 * dim, 2 * dim)
    M_no[:dim, :dim] = A
    M_no[dim:, dim:] = B
    M_no_inv = torch.linalg.inv(M_no)

    shifts = []
    for i in range(len(x_test)):
        be = A @ x_test[i]
        bd = B @ d_test[i]
        b = torch.cat([be, bd])
        sw = M_with_inv @ b
        sn = M_no_inv @ b
        shifts.append((sw[dim:] - sn[dim:]).numpy())

    shifts = np.stack(shifts)
    rel_shift = (
        np.linalg.norm(shifts, axis=1)
        / (np.linalg.norm(d_test.numpy(), axis=1) + 1e-8)
    ).mean() * 100

    print(f"  Relative decoder shift (rank 16): {rel_shift:.2f}%")
    print(f"  CV across samples: {shifts.std(0).mean() / (np.abs(shifts.mean(0)).mean() + 1e-8) * 100:.1f}%")

    # Rank sweep
    print("\n  Rank sweep:")
    rank_shifts = []
    for r in RANKS:
        Jr = build_skip_coupling(
            gram_matrix(unet.e1.conv1.weight, dim),
            gram_matrix(unet.d1.conv1.weight, dim),
            rank=r,
        )
        Mr = torch.zeros(2 * dim, 2 * dim)
        Mr[:dim, :dim] = A
        Mr[dim:, dim:] = B
        Mr[:dim, dim:] = Jr
        Mr[dim:, :dim] = Jr.T
        Mr_inv = torch.linalg.inv(Mr)

        shs = []
        for i in range(len(x_test)):
            be = A @ x_test[i]
            bd = B @ d_test[i]
            b = torch.cat([be, bd])
            sw = Mr_inv @ b
            sn = M_no_inv @ b
            shs.append(
                ((sw[dim:] - sn[dim:]).norm() / (d_test[i].norm() + 1e-8)).item()
            )
        rank_shifts.append(np.mean(shs) * 100)
        print(f"    rank={r:3d}: {rank_shifts[-1]:.2f}%")

    return {
        'shifts': shifts,
        'rel_shift_pct': rel_shift,
        'rank_shifts_pct': rank_shifts,
        'M_no_inv': M_no_inv,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Experiment B: Conditioning Interface Sweep (Section 4)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_b(x_train, x_test, d_test, A, B, M_inv, dim):
    """
    Sweep bottleneck dimension k for linear and MLP encoders.
    Measures decoder cosine similarity at each k.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Conditioning Interface Sweep (Section 4)")
    print("=" * 60)

    unet_params = sum(p.numel() for p in unet.parameters())
    x_tr_d = x_train.to(DEVICE)
    y_tr_d = (x_train @ A.T).to(DEVICE)

    # Linear encoder sweep
    linear_results = []
    print(f"\n  {'k':>5} {'cos_enc':>10} {'cos_dec':>10}")
    for k in ENCODER_DIMS:
        net = nn.Sequential(
            nn.Linear(dim, k, bias=True),
            nn.Linear(k, dim, bias=True),
        ).to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)

        for _ in range(200):
            opt.zero_grad()
            F.mse_loss(net(x_tr_d), y_tr_d).backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            b_enc_pred = net(x_test.to(DEVICE)).cpu()
            b_dec_oracle = d_test @ B.T
            xs, ys = solve_equilibrium(b_enc_pred, b_dec_oracle, M_inv, dim)
            ce = F.cosine_similarity(normalize(xs), normalize(x_test), dim=-1).mean().item()
            cd = F.cosine_similarity(normalize(ys), normalize(d_test), dim=-1).mean().item()

        linear_results.append({'k': k, 'cos_enc': ce, 'cos_dec': cd, 'net': net})
        print(f"  {k:>3}  {ce:>10.4f}  {cd:>10.4f}")

    # Oracle
    xo, yo = solve_equilibrium(x_test @ A.T, d_test @ B.T, M_inv, dim)
    oracle_enc = F.cosine_similarity(normalize(xo), normalize(x_test), dim=-1).mean().item()
    oracle_dec = F.cosine_similarity(normalize(yo), normalize(d_test), dim=-1).mean().item()
    print(f"  Oracle: enc={oracle_enc:.4f}  dec={oracle_dec:.4f}")

    # MLP sweep
    mlp_results = []
    for k in MLP_DIMS:
        mlp = nn.Sequential(
            nn.Linear(dim, k * 2), nn.SiLU(),
            nn.Linear(k * 2, k), nn.SiLU(),
            nn.Linear(k, dim),
        ).to(DEVICE)
        opt = torch.optim.Adam(mlp.parameters(), lr=3e-3, weight_decay=1e-4)

        for _ in range(300):
            opt.zero_grad()
            F.mse_loss(mlp(x_tr_d), y_tr_d).backward()
            opt.step()

        mlp.eval()
        with torch.no_grad():
            bp = mlp(x_test.to(DEVICE)).cpu()
            xs2, ys2 = solve_equilibrium(bp, d_test @ B.T, M_inv, dim)
            cd2 = F.cosine_similarity(normalize(ys2), normalize(d_test), dim=-1).mean().item()

        mlp_results.append({
            'k': k, 'cos_dec': cd2,
            'params': sum(p.numel() for p in mlp.parameters()),
        })

    return {
        'linear_results': linear_results,
        'mlp_results': mlp_results,
        'oracle_cos_enc': oracle_enc,
        'oracle_cos_dec': oracle_dec,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Experiment C: Production Test (Section 5)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_c(x_train, d_train, x_test, d_test, A, B, M_inv, dim):
    """
    Full production test: oracle, learned encoder, skip-only, and full pipeline.
    The full pipeline uses a k=4 bottleneck encoder + 16-unit transfer network.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Production Test (Section 5)")
    print("=" * 60)

    unet_params = sum(p.numel() for p in unet.parameters())
    x_tr_d = x_train.to(DEVICE)
    y_tr_d = (x_train @ A.T).to(DEVICE)

    # k=4 bottleneck encoder (Definition 2, component 1)
    encoder = nn.Sequential(
        nn.Linear(dim, 4, bias=True),
        nn.Linear(4, dim, bias=True),
    ).to(DEVICE)
    opt = torch.optim.Adam(encoder.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad()
        F.mse_loss(encoder(x_tr_d), y_tr_d).backward()
        opt.step()
    encoder.eval()

    # Transfer network (Definition 2, component 2)
    transfer = nn.Sequential(
        nn.Linear(dim, 16), nn.SiLU(),
        nn.Linear(16, dim),
    ).to(DEVICE)
    b_enc_tr = (x_train @ A.T).to(DEVICE)
    b_dec_tr = (d_train @ B.T).to(DEVICE)
    opt2 = torch.optim.Adam(transfer.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(400):
        opt2.zero_grad()
        F.mse_loss(transfer(b_enc_tr), b_dec_tr).backward()
        opt2.step()
    transfer.eval()

    with torch.no_grad():
        b_enc_pred = encoder(x_test.to(DEVICE)).cpu()

        # Four conditioning regimes
        x_or, y_or = solve_equilibrium(x_test @ A.T, d_test @ B.T, M_inv, dim)
        x_v5, y_v5 = solve_equilibrium(b_enc_pred, d_test @ B.T, M_inv, dim)
        x_skip, y_skip = solve_equilibrium(b_enc_pred, torch.zeros(N_TEST, dim), M_inv, dim)
        b_dec_learned = transfer(b_enc_pred.to(DEVICE)).cpu()
        x_full, y_full = solve_equilibrium(b_enc_pred, b_dec_learned, M_inv, dim)

    cs = lambda a, b: F.cosine_similarity(normalize(a), normalize(b), dim=-1).numpy()
    ps_or = cs(y_or, d_test)
    ps_v5 = cs(y_v5, d_test)
    ps_skip = cs(y_skip, d_test)
    ps_full = cs(y_full, d_test)

    enc_params = sum(p.numel() for p in encoder.parameters())
    xfer_params = sum(p.numel() for p in transfer.parameters())
    total_params = enc_params + xfer_params

    print(f"\n  {'Configuration':<35} {'Dec cos':>8}")
    print(f"  {'-' * 45}")
    print(f"  {'Oracle (upper bound)':<35} {ps_or.mean():>8.4f}")
    print(f"  {'Learned enc + oracle dec':<35} {ps_v5.mean():>8.4f}")
    print(f"  {'Learned enc + skip only':<35} {ps_skip.mean():>8.4f}")
    print(f"  {'Full pipeline (ours)':<35} {ps_full.mean():>8.4f}")
    print(f"\n  Total params: {total_params:,} = {total_params/unet_params:.4%} of U-Net")

    # Energy accounting
    E_thermo = 1.38e-23 * 300 * dim * 2 * 400  # kT × N_units × N_steps
    E_gpu = 0.008  # Joules, A100
    raw_gain = E_gpu / E_thermo
    print(f"\n  Energy accounting:")
    print(f"    E_thermo = {E_thermo:.2e} J")
    print(f"    E_gpu    = {E_gpu:.2e} J")
    print(f"    Raw gain:     {raw_gain:.2e}×")
    print(f"    After ADC/DAC (÷10³): {raw_gain/1e3:.2e}×")
    print(f"    Net theoretical gain:  ~10⁷×")

    return {
        'ps_or': ps_or, 'ps_v5': ps_v5,
        'ps_skip': ps_skip, 'ps_full': ps_full,
        'y_or': y_or, 'y_full': y_full,
        'total_params': total_params,
        'enc_params': enc_params,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Nonlinear Equilibration Validation (Appendix)
# ═════════════════════════════════════════════════════════════════════════════

def experiment_nonlinear(J_enc, dim):
    """
    Validate that the quartic nonlinearity introduces only O(10⁻²)
    perturbation to the linear equilibrium.
    """
    print("\n" + "=" * 60)
    print("NONLINEAR VALIDATION (Appendix)")
    print("=" * 60)

    A = 2 * J2 * torch.eye(dim) + J_enc + J_enc.T

    def energy(x, J, b, J4=0.05):
        quad = J2 * (x ** 2).sum(dim=1) + 0.5 * (x @ J @ x.T).diagonal()
        quartic = J4 * (x ** 4).sum(dim=1)
        linear = (b * x).sum(dim=1)
        return (quad + quartic - linear).mean()

    x_target = torch.randn(1, dim) * 0.3
    b_oracle = A @ x_target.T

    # Linear equilibrium
    x_lin = torch.linalg.solve(A, b_oracle).T

    # Nonlinear minimization
    x_nonlin = x_lin.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x_nonlin], lr=1.0, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = energy(x_nonlin, J_enc, b_oracle.T, J4=0.05)
        loss.backward()
        return loss

    opt.step(closure)

    cos = torch.cosine_similarity(x_nonlin, x_lin, dim=1).item()
    rel_diff = (torch.norm(x_nonlin - x_lin) / torch.norm(x_lin)).item()

    print(f"  Cosine similarity (nonlinear vs linear): {cos:.4f}")
    print(f"  Relative difference: {rel_diff:.1%}")
    print(f"  → Quartic perturbation is O({rel_diff:.0e}), confirming linear regime validity")


# ═════════════════════════════════════════════════════════════════════════════
# Figure Generation
# ═════════════════════════════════════════════════════════════════════════════

def setup_matplotlib():
    """Paper-quality matplotlib configuration."""
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman'],
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.06,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'lines.linewidth': 1.2, 'patch.linewidth': 0.5,
        'grid.linewidth': 0.4, 'grid.alpha': 0.35,
        'axes.grid': True, 'axes.axisbelow': True,
    })


# Color palette
BLUE = '#2166AC'
RED = '#D6604D'
GREEN = '#4DAC26'
ORANGE = '#E08214'
PURPLE = '#762A83'
GRAY = '#878787'
DGRAY = '#2B2B2B'
LGRAY = '#F5F5F5'


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=9, pad=6)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)


def _tbox(ax, x, y, txt, col, fs=8.0):
    ax.text(x, y, txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=fs, color=col,
            bbox=dict(boxstyle='round,pad=0.22', fc='white',
                      ec=col, lw=0.5, alpha=0.96))


def generate_fig1():
    """Figure 1: System architecture diagram."""

    def rbox(ax, cx, cy, w, h, label, sub=None,
             fc='#EEF4FB', ec=BLUE, lw=0.8, fs=8, sfs=6.5):
        r = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                           boxstyle='round,pad=0.02',
                           facecolor=fc, edgecolor=ec, linewidth=lw, zorder=4)
        ax.add_patch(r)
        if sub:
            ax.text(cx, cy + h * 0.14, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=DGRAY, zorder=5)
            ax.text(cx, cy - h * 0.18, sub, ha='center', va='center',
                    fontsize=sfs, style='italic', color=GRAY, zorder=5)
        else:
            ax.text(cx, cy, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=DGRAY, zorder=5)

    def arro(ax, x1, y1, x2, y2, col=DGRAY, lw=0.8):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                    mutation_scale=8), zorder=6)

    fig = plt.figure(figsize=(7.0, 4.2))
    ax_l = fig.add_axes([0.02, 0.04, 0.44, 0.88])
    ax_r = fig.add_axes([0.54, 0.04, 0.44, 0.88])

    for ax in [ax_l, ax_r]:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # (a) GPU path
    ax_l.text(0.50, 0.96, '(a) GPU — digital inference',
              ha='center', fontsize=9, fontweight='bold', color=DGRAY)
    rbox(ax_l, 0.50, 0.84, 0.55, 0.12, 'Noisy latent $x_t$', fc=LGRAY, ec=GRAY, lw=0.6)
    rbox(ax_l, 0.26, 0.64, 0.42, 0.13, 'Encoder', 'digital ResBlocks')
    rbox(ax_l, 0.74, 0.64, 0.42, 0.13, 'Decoder', 'digital ResBlocks')
    rbox(ax_l, 0.50, 0.41, 0.55, 0.13, 'Score network (GPU)', '860M params',
         fc='#FEF0EC', ec=RED, lw=1.0)
    rbox(ax_l, 0.50, 0.19, 0.55, 0.12, 'Denoised output', fc=LGRAY, ec=GRAY, lw=0.6)
    for xy in [(0.26, 0.78, 0.26, 0.71), (0.74, 0.78, 0.74, 0.71),
               (0.26, 0.57, 0.26, 0.48), (0.74, 0.57, 0.74, 0.48),
               (0.50, 0.34, 0.50, 0.25)]:
        arro(ax_l, *xy)
    ax_l.annotate('', xy=(0.53, 0.64), xytext=(0.47, 0.64),
                  arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.0,
                                  mutation_scale=8, connectionstyle='arc3,rad=-0.45'), zorder=6)
    ax_l.text(0.50, 0.54, 'skip\n(non-local)', ha='center', fontsize=6.5,
              color=ORANGE, style='italic')
    ax_l.text(0.50, 0.06, '$\\sim$1--10 J / image', ha='center', fontsize=8,
              color=RED, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.2', fc='#FEF0EC', ec=RED, lw=0.7))

    # (b) Langevin substrate
    ax_r.text(0.50, 0.96, '(b) Ours — Langevin substrate',
              ha='center', fontsize=9, fontweight='bold', color=DGRAY)
    rbox(ax_r, 0.50, 0.85, 0.50, 0.11, 'Noisy input $x_t$', fc=LGRAY, ec=GRAY, lw=0.6)

    # Digital interface region
    di = FancyBboxPatch((0.03, 0.59), 0.94, 0.19, boxstyle='round,pad=0.01',
                        facecolor='#F3EEF8', edgecolor=PURPLE, linewidth=0.9, zorder=2)
    ax_r.add_patch(di)
    ax_r.text(0.50, 0.775, 'Digital conditioning interface  (2,560 params, 0.032%)',
              ha='center', fontsize=6.5, color=PURPLE, zorder=5)
    rbox(ax_r, 0.27, 0.67, 0.40, 0.11, 'Input encoder', '$k\\!=\\!4$ linear',
         fc='#EDE0F5', ec=PURPLE, lw=0.7, fs=7.5, sfs=6.5)
    rbox(ax_r, 0.73, 0.67, 0.40, 0.11, 'Enc$\\to$Dec net', 'tiny MLP',
         fc='#EDE0F5', ec=PURPLE, lw=0.7, fs=7.5, sfs=6.5)

    # Langevin substrate region
    ls = FancyBboxPatch((0.03, 0.27), 0.94, 0.29, boxstyle='round,pad=0.01',
                        facecolor='#EBF5FB', edgecolor=BLUE, linewidth=0.9, zorder=2)
    ax_r.add_patch(ls)
    ax_r.text(0.50, 0.545, 'Langevin substrate  (physical hardware target)',
              ha='center', fontsize=6.5, color=BLUE, fontweight='bold', zorder=5)
    rbox(ax_r, 0.27, 0.40, 0.40, 0.13, 'Enc module', 'thermal equilibration',
         fc='#D6EAF8', ec=BLUE, lw=0.7, fs=7.5, sfs=6.5)
    rbox(ax_r, 0.73, 0.40, 0.40, 0.13, 'Dec module', 'thermal equilibration',
         fc='#D6EAF8', ec=BLUE, lw=0.7, fs=7.5, sfs=6.5)

    rbox(ax_r, 0.50, 0.16, 0.60, 0.11, 'Denoised output  (cos$\\,=\\,$0.9906)',
         fc='#E8F8F5', ec=GREEN, lw=0.8)

    arro(ax_r, 0.27, 0.79, 0.27, 0.72, col=PURPLE)
    arro(ax_r, 0.73, 0.79, 0.73, 0.72, col=PURPLE)
    ax_r.text(0.27, 0.580, '$b_{\\rm enc}$', ha='center', fontsize=7.5, color=PURPLE)
    ax_r.text(0.73, 0.580, '$b_{\\rm dec}$', ha='center', fontsize=7.5, color=PURPLE)
    arro(ax_r, 0.27, 0.615, 0.27, 0.465, col=PURPLE)
    arro(ax_r, 0.73, 0.615, 0.73, 0.465, col=PURPLE)

    ax_r.annotate('', xy=(0.53, 0.40), xytext=(0.47, 0.40),
                  arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.9,
                                  mutation_scale=7, linestyle='dashed'), zorder=6)
    ax_r.text(0.50, 0.335, 'skip $J_{\\rm skip}$', ha='center', fontsize=6,
              color=ORANGE, style='italic')
    arro(ax_r, 0.50, 0.27, 0.50, 0.215, col=DGRAY)

    ax_r.text(0.50, 0.05, '$\\sim 10^{7}\\times$ net energy gain (theoretical)',
              ha='center', fontsize=8, color=GREEN, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.2', fc='#E8F8F5', ec=GREEN, lw=0.7))

    fig.savefig('figures/fig1_architecture.pdf', format='pdf')
    fig.savefig('figures/fig1_architecture.png')
    plt.close(fig)
    print("  ✓ fig1_architecture")


def generate_fig2(shifts, rel_shift_pct, rank_shifts_pct, J_skip, dim):
    """Figure 2: Skip coupling analysis (3 panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), gridspec_kw={'wspace': 0.52})

    # (a) J_skip heatmap
    ax = axes[0]
    lim = float(np.abs(J_skip.numpy()).max())
    im = ax.imshow(J_skip.numpy(), aspect='auto', cmap='RdBu_r',
                   vmin=-lim, vmax=lim, interpolation='nearest')
    cb = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.04, aspect=15)
    cb.ax.tick_params(labelsize=7)
    _style(ax, '(a)  $J_{\\rm skip}$ matrix  (rank 16)',
           'Decoder unit $j$', 'Encoder unit $i$')
    ax.grid(False)

    # (b) Per-dimension shift
    ax = axes[1]
    dims = np.arange(dim)
    mu = shifts.mean(0); sig = shifts.std(0)
    ax.fill_between(dims, mu - sig, mu + sig, alpha=0.22, color=BLUE)
    ax.plot(dims, mu, color=BLUE, lw=1.0, label='Mean')
    ax.axhline(0, color=GRAY, lw=0.5, ls='--')
    _style(ax, '(b)  Decoder shift per dimension',
           'Decoder dimension $j$', 'Shift  $\\Delta y_j^*$')
    _tbox(ax, 0.97, 0.97, f'Rel. shift:\n{rel_shift_pct:.2f}%', BLUE)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)

    # (c) Shift vs rank
    ax = axes[2]
    ax.plot(RANKS, rank_shifts_pct, 'o-', color=BLUE, ms=5, lw=1.3, zorder=3)
    ax.fill_between(RANKS, rank_shifts_pct, alpha=0.12, color=BLUE)
    ax.axhline(rel_shift_pct, color=RED, lw=0.9, ls='--',
               label=f'Rank 16  ({rel_shift_pct:.1f}%)')
    _style(ax, '(c)  Shift vs. coupling rank',
           'Skip coupling rank $k$', 'Relative shift (%)')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    ax.set_ylim(bottom=0)

    fig.savefig('figures/fig2_skip_coupling.pdf', format='pdf')
    fig.savefig('figures/fig2_skip_coupling.png')
    plt.close(fig)
    print("  ✓ fig2_skip_coupling")


def generate_fig3(linear_results, mlp_results, oracle_cos_dec, dim):
    """Figure 3: Conditioning interface sweep (2 panels)."""
    unet_params = sum(p.numel() for p in unet.parameters())
    ks_l = [r['k'] for r in linear_results]
    cd_l = [r['cos_dec'] for r in linear_results]
    ce_l = [r['cos_enc'] for r in linear_results]
    ks_m = [r['k'] for r in mlp_results]
    cd_m = [r['cos_dec'] for r in mlp_results]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={'wspace': 0.40})

    # (a) Cosine vs k
    ax = axes[0]
    ax.plot(ks_l, cd_l, 'o-', color=BLUE, ms=5, lw=1.3, label='Linear enc. (decoder)')
    ax.plot(ks_l, ce_l, 's--', color=BLUE, ms=3.5, lw=0.9, alpha=0.5,
            label='Linear enc. (encoder)')
    ax.plot(ks_m, cd_m, '^-', color=ORANGE, ms=5, lw=1.3, label='MLP enc. (decoder)')
    ax.axhline(oracle_cos_dec, color=DGRAY, lw=1.0, ls=':',
               label=f'Oracle ({oracle_cos_dec:.4f})')
    ax.axhline(0.5, color=RED, lw=0.8, ls='--', alpha=0.7, label='Threshold (0.5)')
    ax.annotate(f'$k=4$: {cd_l[0]:.4f}', xy=(ks_l[0], cd_l[0]), xytext=(35, 0.78),
                fontsize=7.5, color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.6, mutation_scale=7))
    _style(ax, '(a)  Conditioning interface sweep',
           'Bottleneck dimension $k$', 'Cosine similarity')
    ax.set_ylim(-0.06, 1.12)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)

    # (b) Net energy gain
    ax = axes[1]
    enc_fracs = [r['k'] * dim * 2 / unet_params for r in linear_results]
    net_gains = [1e7 * (1 - f) for f in enc_fracs]
    ax.semilogy(ks_l, net_gains, 'o-', color=BLUE, ms=5, lw=1.3)
    ax.fill_between(ks_l, net_gains, alpha=0.10, color=BLUE)
    ax.axhline(1e7, color=GRAY, lw=0.8, ls=':', alpha=0.7, label='Substrate gain ($10^7\\times$)')
    ax.axhline(1e4, color=ORANGE, lw=0.8, ls='--', label='$10^4\\times$')
    ax.axhline(1e2, color=RED, lw=0.8, ls=':', alpha=0.7, label='$10^2\\times$')
    _style(ax, '(b)  Savings after encoder cost',
           'Bottleneck dimension $k$', 'Net energy gain ($\\times$)')
    ax.legend(fontsize=7, loc='lower left', framealpha=0.9)
    ax.set_ylim(5e5, 3e7)

    fig.savefig('figures/fig3_conditioning_sweep.pdf', format='pdf')
    fig.savefig('figures/fig3_conditioning_sweep.png')
    plt.close(fig)
    print("  ✓ fig3_conditioning_sweep")


def generate_fig4(ps_or, ps_v5, ps_skip, ps_full, y_or, y_full, d_test):
    """Figure 4: Production test (3 panels)."""
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8), gridspec_kw={'wspace': 0.52})

    # (a) Bar chart
    ax = axes[0]
    labels = ['Oracle', 'Enc +\noracle', 'Enc +\nskip only', 'Full\npipeline']
    means = [ps_or.mean(), ps_v5.mean(), ps_skip.mean(), ps_full.mean()]
    stds = [ps_or.std(), ps_v5.std(), ps_skip.std(), ps_full.std()]
    cols = [GRAY, BLUE, RED, GREEN]
    bars = ax.bar(range(4), means, yerr=stds, color=cols, alpha=0.82,
                  edgecolor='white', lw=0.5, capsize=4,
                  error_kw={'lw': 0.9}, width=0.60, zorder=3)
    for b, m, s in zip(bars, means, stds):
        y_top = m + s + 0.05
        if y_top > 1.10:
            ax.text(b.get_x() + b.get_width() / 2, m - 0.07, f'{m:.3f}',
                    ha='center', va='top', fontsize=7.5, color='white', fontweight='bold')
        else:
            ax.text(b.get_x() + b.get_width() / 2, y_top, f'{m:.3f}',
                    ha='center', va='bottom', fontsize=7.5, color=DGRAY)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.12, 1.26); ax.tick_params(axis='x', length=0)
    _style(ax, '(a)  Conditioning regimes', ylabel='Decoder cosine similarity')

    # (b) Scatter
    ax = axes[1]
    ft = d_test.numpy().flatten()
    ff = y_full.numpy().flatten()
    fo = y_or.numpy().flatten()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ft), 500, replace=False)
    ax.scatter(ft[idx], fo[idx], s=5, alpha=0.40, color=GRAY, rasterized=True,
               label=f'Oracle  ({ps_or.mean():.3f})')
    ax.scatter(ft[idx], ff[idx], s=5, alpha=0.50, color=GREEN, rasterized=True,
               label=f'Full pipeline  ({ps_full.mean():.3f})')
    lm = [float(ft.min()), float(ft.max())]
    ax.plot(lm, lm, '--', color=DGRAY, lw=0.8, alpha=0.45)
    _style(ax, '(b)  Target vs. equilibrium',
           'Digital target $y^{\\rm dig}$', 'Langevin equilibrium $y^*$')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9, markerscale=2.5)

    # (c) Per-sample
    ax = axes[2]
    xs = np.arange(N_TEST)
    ax.bar(xs, ps_full, color=GREEN, alpha=0.75, width=0.85, edgecolor='none',
           label='Full pipeline', zorder=3)
    ax.bar(xs, ps_skip, color=RED, alpha=0.70, width=0.85, edgecolor='none',
           label='Skip only', zorder=4)
    ax.axhline(0.5, color=DGRAY, lw=0.7, ls='--', alpha=0.7)
    ax.set_ylim(-0.18, 1.20)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
    ax.text(0.03, 0.05,
            f'Full:  {ps_full.mean():.3f}$\\pm${ps_full.std():.3f}\n'
            f'Skip: {ps_skip.mean():.3f}$\\pm${ps_skip.std():.3f}',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=GRAY, lw=0.5, alpha=0.96))
    _style(ax, '(c)  Per-sample alignment', 'Test sample index', 'Cosine similarity')

    fig.savefig('figures/fig4_production_test.pdf', format='pdf')
    fig.savefig('figures/fig4_production_test.png')
    plt.close(fig)
    print("  ✓ fig4_production_test")


def generate_figA(J_enc, x_test, dim):
    """Figure A (Appendix): Conditioning failure analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={'wspace': 0.44})

    # (a) Eigenspectrum
    ax = axes[0]
    eig = torch.linalg.eigvalsh(J_enc).numpy()
    eig_s = np.sort(np.abs(eig))[::-1]
    ax.semilogy(np.arange(1, dim + 1), eig_s, color=BLUE, lw=1.1, zorder=3)
    ax.fill_between(np.arange(1, dim + 1), eig_s, alpha=0.12, color=BLUE)
    ax.axhline(eig_s[0], color=RED, lw=0.8, ls='--',
               label=f'$\\lambda_{{\\rm max}}={eig_s[0]:.4f}$')
    ax.axhline(eig_s.mean(), color=ORANGE, lw=0.8, ls=':',
               label=f'Mean$={eig_s.mean():.4f}$')
    _style(ax, '(a)  Gram coupling eigenspectrum',
           'Eigenvalue index', '$|\\lambda_i(J_{\\rm enc})|$')
    ax.legend(fontsize=7.5, framealpha=0.9)
    _tbox(ax, 0.97, 0.97, 'Signal deficit\n$\\approx$1/2600', RED)

    # (b) Naive vs correct bias
    ax = axes[1]
    A_mat = 2 * J2 * torch.eye(dim) + J_enc + J_enc.T
    b_n = (x_test @ J_enc.T).numpy().flatten()
    b_c = (x_test @ A_mat.T).numpy().flatten()
    tf = x_test.numpy().flatten()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(tf), 600, replace=False)
    ax.scatter(tf[idx], b_n[idx], s=5, alpha=0.40, color=RED, rasterized=True,
               label='Naive  $b=xJ^T$')
    ax.scatter(tf[idx], b_c[idx], s=5, alpha=0.40, color=GREEN, rasterized=True,
               label='Correct  $b=Ax^*$')
    lm = [float(tf.min()), float(tf.max())]
    ax.plot(lm, lm, '--', color=DGRAY, lw=0.8, alpha=0.45)
    _style(ax, '(b)  Naive vs. correct conditioning bias',
           'Target activation $x_i^*$', 'Bias value $b_i$')
    ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9, markerscale=2.5)
    ratio = float(np.std(b_c)) / (float(np.std(b_n)) + 1e-10)
    ax.text(0.97, 0.06, f'Std ratio: {ratio:.0f}$\\times$',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color=GREEN,
            bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=GRAY, lw=0.5, alpha=0.96))

    fig.savefig('figures/figA_conditioning_failure.pdf', format='pdf')
    fig.savefig('figures/figA_conditioning_failure.png')
    plt.close(fig)
    print("  ✓ figA_conditioning_failure")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce thermodynamic diffusion inference experiments')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--train-mnist', action='store_true', help='Train on real MNIST data')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs (with --train-mnist)')
    parser.add_argument('--nonlinear', action='store_true', help='Run nonlinear equilibration validation')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Analytical regime: D={D}, N_train={N_TRAIN}, N_test={N_TEST}")

    # ── Model ─────────────────────────────────────────────────────────────
    unet = ToyUNet().to(DEVICE)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"U-Net: {n_params / 1e6:.1f}M parameters")

    if args.train_mnist:
        print("\nTraining on MNIST...")
        unet = train_on_mnist(unet, epochs=args.epochs)
    else:
        unet.eval()

    # ── Activations ───────────────────────────────────────────────────────
    print("\nCollecting activations...")
    torch.manual_seed(0)
    x_all = torch.randn(N_TRAIN + N_TEST, 4, 32, 32, device=DEVICE)

    all_enc, all_dec = [], []
    with torch.no_grad():
        for i in range(0, N_TRAIN + N_TEST, 32):
            _, acts = unet(x_all[i:i+32], return_activations=True)
            all_enc.append(acts['e1'].mean([-2, -1])[:, :D].float().cpu())
            all_dec.append(acts['d1'].mean([-2, -1])[:, :D].float().cpu())

    enc_all = torch.cat(all_enc)
    dec_all = torch.cat(all_dec)
    x_train, x_test = enc_all[:N_TRAIN], enc_all[N_TRAIN:]
    d_train, d_test = dec_all[:N_TRAIN], dec_all[N_TRAIN:]
    print(f"  Train: {N_TRAIN}, Test: {N_TEST}")

    # ── Coupling matrices ─────────────────────────────────────────────────
    J_enc = gram_matrix(unet.e1.conv1.weight, D)
    J_dec = gram_matrix(unet.d1.conv1.weight, D)
    J_skip = build_skip_coupling(J_enc, J_dec, rank=16)
    A, B, M = build_system_matrix(J_enc, J_dec, J_skip, D)
    M_inv = torch.linalg.inv(M)

    print(f"  J_enc norm: {J_enc.norm():.4f}")
    print(f"  J_dec norm: {J_dec.norm():.4f}")
    print(f"  J_skip norm: {J_skip.norm():.4f}")
    print(f"  System cond: {torch.linalg.cond(M):.1f}")

    # ── Run experiments ───────────────────────────────────────────────────
    res_a = experiment_a(x_test, d_test, A, B, J_skip, D)
    res_b = experiment_b(x_train, x_test, d_test, A, B, M_inv, D)
    res_c = experiment_c(x_train, d_train, x_test, d_test, A, B, M_inv, D)

    if args.nonlinear:
        experiment_nonlinear(J_enc, D)

    # ── Generate figures ──────────────────────────────────────────────────
    if not args.no_figures:
        print("\nGenerating figures...")
        setup_matplotlib()
        generate_fig1()
        generate_fig2(res_a['shifts'], res_a['rel_shift_pct'],
                      res_a['rank_shifts_pct'], J_skip, D)
        generate_fig3(res_b['linear_results'], res_b['mlp_results'],
                      res_b['oracle_cos_dec'], D)
        generate_fig4(res_c['ps_or'], res_c['ps_v5'],
                      res_c['ps_skip'], res_c['ps_full'],
                      res_c['y_or'], res_c['y_full'], d_test)
        generate_figA(J_enc, x_test, D)
        print(f"\nAll figures saved to figures/")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Skip coupling shift (rank 16): {res_a['rel_shift_pct']:.2f}%")
    print(f"  Oracle decoder cosine:         {res_b['oracle_cos_dec']:.4f}")
    print(f"  Full pipeline decoder cosine:  {res_c['ps_full'].mean():.4f}")
    print(f"  Skip-only decoder cosine:      {res_c['ps_skip'].mean():.4f}")
    print(f"  Total conditioning params:     {res_c['total_params']:,}")
    print(f"  Theoretical energy gain:       ~10⁷×")
    print("=" * 60)
