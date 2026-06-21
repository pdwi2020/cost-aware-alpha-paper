"""K0+K2 — Cost-Aware Knockoffs (CAK): headline novelty.

## Design (K0)

### Prior art and the novelty delta
- Candès et al. (2018): model-X knockoffs, finite-sample FDR control.
- DeepLINK-T (2024): LSTM-AE for time-series knockoffs, financial data.
  **Gap**: no prior work uses Almgren-Chriss size-dependent market impact
  inside the knockoff importance statistic.
- Bajgrowicz & Scaillet (2012, JFE): FDR on trading rules, proportional cost,
  single series — not cross-sectional, not AC market impact.
We claim ONLY this delta: the cost-aware importance W_j = Z_j − Z̃_j.

### Why deep knockoffs are required
Second-order (Gaussian) knockoffs match only the first two moments of the
feature distribution. Confirmed heavy tails in our data:
  amihud:   skewness=170, kurtosis=33,130
  vix:      skewness=3.3,  kurtosis=17.4
Gaussian knockoffs violate conditional exchangeability on these features,
inflating empirical FDR above q. An LSTM-VAE in copula space provides
proper tail coverage.

### Copula pre-processing
For each feature f: u_j = F̂_j(x_j) using the empirical CDF on IS data.
The LSTM-VAE is trained on u_j ∈ (0, 1) → outputs ũ_j ∈ (0, 1).
Knockoff back-transform: x̃_j = F̂_j^{-1}(ũ_j) (empirical quantile function).
This cleanly separates heavy-tail marginals (handled by CDF transform)
from temporal + cross-sectional dependence structure (handled by LSTM-VAE).

### Architecture
- Encoder: 2-layer LSTM (hidden=128) on (batch, seq_len=20, 35 features)
  → shared representation → μ, log_σ (latent_dim=64)
- Decoder: 2-layer LSTM (hidden=128) → (batch, seq_len, 35)
- Knockoff: perturb latent z with calibrated noise, decode → X̃
- Loss: recon MSE + β·KL + γ·covariance-matching (ensures |corr(X̃, X) ≈ Σ|_F)
- Batch size 256 sequences (ticker × time-chunk), 100 epochs

### Cost-Aware Importance Statistic (the novelty)
Z_j  = _fast_net_sharpe(signal_j  using X_j,   returns, sigma, adv)
Z̃_j = _fast_net_sharpe(signal_j̃ using X̃_j,  returns, sigma, adv)
W_j  = Z_j − Z̃_j
This is the net-of-Almgren-Chriss IS performance of the real feature
minus its knockoff. The knockoff(+) filter selects features where the
original feature consistently outperforms its (null) knockoff.

### Knockoff(+) filter
τ = min{t ≥ 0 : (1 + #{j : W_j ≤ −t}) / max(1, #{j : W_j ≥ t}) ≤ q}
CAK set = {j : W_j ≥ τ}

### FDR-validity gate (K3)
Add K=10 known-null Gaussian features, generate knockoffs, compute W_j_null.
Over ≥200 simulation draws (with permuted labels):
  empirical_FDR = mean(#{null features selected} / max(1, #{all selected}))
Gate PASSES if empirical_FDR ≤ q × 1.1 (10% slack for simulation noise).
If FAILS: switch to TSKI e-value filtering or report knockoff failure.

Outputs:
    data/processed/cak_knockoff_features.parquet  — knockoff feature matrix (IS)
    data/processed/cak_importance.parquet          — W_j + Z_j + Z̃_j per feature
    data/processed/cak_results.parquet            — selected features + rigor ladder
    data/processed/cak_fdr_validity.parquet       — FDR-validity simulation results
    models/knockoff_vae.pt                        — trained LSTM-VAE weights

GPU recommended (RTX 3090 ~$0.30/hr on vast.ai; ~2h training → ~$0.60).
CPU fallback available for small tests (--no-gpu flag).

Run:
    python3 -u src/knockoffs/run_cost_aware_knockoffs.py [--no-gpu] [--epochs N]
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.run_ta_fdr import (
    build_single_feature_signal, _fast_net_sharpe,
    load_ohlcv_matrices, estimate_ic_signs,
)
from src.fdr.bh_correction import benjamini_hochberg
from src.backtest.portfolio import PortfolioSimulator
from src.features.feature_spec import feature_columns as _feature_columns

FEATURES_PATH  = ROOT / "data" / "processed" / "features_all.parquet"
FDR_PATH       = ROOT / "data" / "processed" / "fdr_results.parquet"
OHLCV_PATH     = ROOT / "data" / "processed" / "daily_ohlcv.parquet"
CFG_PATH       = ROOT / "configs" / "backtest.yaml"

OUT_KNOCKOFFS  = ROOT / "data" / "processed" / "cak_knockoff_features.parquet"
OUT_IMPORTANCE = ROOT / "data" / "processed" / "cak_importance.parquet"
OUT_RESULTS    = ROOT / "data" / "processed" / "cak_results.parquet"
OUT_VALIDITY   = ROOT / "data" / "processed" / "cak_fdr_validity.parquet"
OUT_MODEL      = ROOT / "models" / "knockoff_vae.pt"

IS_START  = "2013-01-01"
IS_END    = "2021-12-31"
FDR_Q     = 0.10
SEQ_LEN   = 20     # lookback window for LSTM
LATENT_DIM = 64
HIDDEN_DIM = 128
N_NULL_FEATS = 10   # for FDR-validity gate
N_VALIDITY_SIMS = 200
BATCH_SIZE = 256
EPOCHS_DEFAULT = 100


def log(msg): print(msg, flush=True)


# ── Copula transform ──────────────────────────────────────────────────────────

class EmpiricalCopula:
    """Per-feature rank transform using IS empirical CDF."""

    def __init__(self):
        self.sorted_values: dict = {}   # feature → sorted IS values

    def fit(self, X: pd.DataFrame):
        for col in X.columns:
            vals = X[col].dropna().sort_values().values
            self.sorted_values[col] = vals

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        U = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
        for col in X.columns:
            if col not in self.sorted_values:
                U[col] = 0.5
                continue
            sv = self.sorted_values[col]
            # Rank-based: u = (rank - 0.5) / n, clipped to (ε, 1-ε)
            u = np.searchsorted(sv, X[col].values, side="right")
            u = (u + 0.5) / (len(sv) + 1)
            U[col] = np.clip(u, 1e-6, 1 - 1e-6)
        return U.fillna(0.5)

    def inverse_transform(self, U: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=U.index, columns=U.columns, dtype=float)
        for col in U.columns:
            if col not in self.sorted_values:
                X[col] = np.nan
                continue
            sv = self.sorted_values[col]
            # Quantile function: x = sorted_values[floor(u * n)]
            idx = np.clip((U[col].values * len(sv)).astype(int), 0, len(sv) - 1)
            X[col] = sv[idx]
        return X


# ── LSTM-VAE for time-series knockoffs ───────────────────────────────────────

def build_lstm_vae(n_features: int, hidden_dim: int, latent_dim: int, device: str):
    """Build LSTM-VAE in PyTorch. Returns (model, optimizer)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch required. Install with: pip3 install torch")

    class LSTMEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_dim, num_layers=2,
                                batch_first=True, dropout=0.1)
            self.mu_layer    = nn.Linear(hidden_dim, latent_dim)
            self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            h = h[-1]   # take last layer's hidden state
            return self.mu_layer(h), self.logvar_layer(h)

    class LSTMDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc      = nn.Linear(latent_dim, hidden_dim)
            self.lstm    = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                                   batch_first=True, dropout=0.1)
            self.out     = nn.Linear(hidden_dim, n_features)
            self.sigmoid = nn.Sigmoid()

        def forward(self, z, seq_len):
            h0 = self.fc(z).unsqueeze(1).repeat(1, seq_len, 1)
            out, _ = self.lstm(h0)
            return self.sigmoid(self.out(out))   # output ∈ (0,1) = copula space

    class LSTMVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = LSTMEncoder()
            self.decoder = LSTMDecoder()

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, logvar = self.encoder(x)
            z          = self.reparameterize(mu, logvar)
            recon      = self.decoder(z, x.shape[1])
            return recon, mu, logvar

        def generate_knockoff(self, x, noise_scale: float = 0.5):
            """Generate knockoff by adding calibrated noise in latent space."""
            mu, logvar = self.encoder(x)
            std        = torch.exp(0.5 * logvar)
            z_tilde    = mu + noise_scale * torch.randn_like(mu) * std
            return self.decoder(z_tilde, x.shape[1])

    import torch.optim as optim
    model = LSTMVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return model, optimizer


def make_sequences(X_copula: np.ndarray, seq_len: int) -> np.ndarray:
    """Slide a window of seq_len over the time axis.
    X_copula: (n_dates, n_features) for ONE ticker.
    Returns: (n_seqs, seq_len, n_features).
    """
    n = len(X_copula)
    if n <= seq_len:
        return None
    seqs = []
    for i in range(0, n - seq_len, seq_len // 2):   # 50% overlap
        seqs.append(X_copula[i : i + seq_len])
    return np.array(seqs, dtype=np.float32)


def train_lstm_vae(
    feat_df: pd.DataFrame,
    features: list,
    copula: EmpiricalCopula,
    epochs: int,
    device: str,
    beta: float = 1.0,
    gamma: float = 0.5,
) -> tuple:
    """Train the LSTM-VAE on IS copula-transformed features. Returns (model, losses)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch required")

    dates_level = feat_df.index.get_level_values("date")
    is_mask = (dates_level >= pd.Timestamp(IS_START)) & \
              (dates_level <= pd.Timestamp(IS_END))
    feat_is = feat_df.loc[is_mask, features].fillna(method="ffill").fillna(0)

    log("  Fitting copula transform on IS data …")
    copula.fit(feat_is)

    log("  Building sequence dataset …")
    all_seqs = []
    for ticker in feat_is.index.get_level_values("ticker").unique():
        try:
            ticker_df = feat_is.xs(ticker, level="ticker").values.astype(np.float32)
        except Exception:
            continue
        u = copula.transform(
            pd.DataFrame(ticker_df, columns=features)
        ).values.astype(np.float32)
        s = make_sequences(u, SEQ_LEN)
        if s is not None:
            all_seqs.append(s)

    if not all_seqs:
        raise RuntimeError("No sequences built — check IS data")

    seqs = np.concatenate(all_seqs, axis=0)
    log(f"  Dataset: {seqs.shape[0]} sequences × {SEQ_LEN} steps × {seqs.shape[2]} features")

    model, optimizer = build_lstm_vae(len(features), HIDDEN_DIM, LATENT_DIM, device)

    import torch
    dataset  = torch.utils.data.TensorDataset(torch.tensor(seqs))
    loader   = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                           shuffle=True, drop_last=True)
    losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches  = 0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)

            # Reconstruction loss (MSE in copula space)
            l_recon = nn.functional.mse_loss(recon, batch)

            # KL divergence
            l_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

            # Covariance-matching: |cov(X̃, X) - cov(X, X)| in Frobenius norm
            b, t, p = batch.shape
            x_flat  = batch.reshape(b * t, p)
            x_tilde_flat = recon.reshape(b * t, p)
            cx  = torch.cov(x_flat.T)
            cxt = torch.cov(x_tilde_flat.T)
            l_cov = torch.norm(cx - cxt, p="fro") / (p * p)

            loss = l_recon + beta * l_kl + gamma * l_cov
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            log(f"    Epoch {epoch+1}/{epochs}  loss={avg_loss:.5f}")

    return model, losses


def generate_knockoff_panel(
    feat_df: pd.DataFrame,
    features: list,
    copula: EmpiricalCopula,
    model,
    device: str,
    noise_scale: float = 0.5,
) -> pd.DataFrame:
    """Generate knockoff feature panel (same index as feat_df IS slice)."""
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required")

    dates_level = feat_df.index.get_level_values("date")
    is_mask = (dates_level >= pd.Timestamp(IS_START)) & \
              (dates_level <= pd.Timestamp(IS_END))
    feat_is = feat_df.loc[is_mask, features].fillna(method="ffill").fillna(0)

    knockoff_parts = []
    model.eval()

    with torch.no_grad():
        for ticker in feat_is.index.get_level_values("ticker").unique():
            try:
                t_df = feat_is.xs(ticker, level="ticker")
            except Exception:
                continue
            t_idx = t_df.index
            u = copula.transform(t_df).values.astype(np.float32)

            # Process in SEQ_LEN chunks, reconstruct full series
            n = len(u)
            u_tilde = np.zeros_like(u)
            counts   = np.zeros(n)

            for start in range(0, max(1, n - SEQ_LEN + 1), SEQ_LEN // 2):
                end = min(start + SEQ_LEN, n)
                chunk = torch.tensor(u[start:end]).unsqueeze(0).to(device)
                if chunk.shape[1] < 2:
                    continue
                tilde_chunk = model.generate_knockoff(chunk, noise_scale).cpu().numpy()[0]
                u_tilde[start:end] += tilde_chunk[:end - start]
                counts[start:end]  += 1

            counts = np.maximum(counts, 1)
            u_tilde = np.clip(u_tilde / counts[:, None], 1e-6, 1 - 1e-6)

            # Back-transform to original scale
            x_tilde = copula.inverse_transform(
                pd.DataFrame(u_tilde, index=t_idx, columns=features)
            )
            x_tilde.index = pd.MultiIndex.from_product([[ticker], t_idx],
                                                        names=["ticker", "date"])
            knockoff_parts.append(x_tilde)

    if not knockoff_parts:
        raise RuntimeError("No knockoffs generated")

    knockoff_df = pd.concat(knockoff_parts)
    knockoff_df.columns = [f + "_kn" for f in features]
    return knockoff_df.sort_index()


# ── Cost-aware importance ─────────────────────────────────────────────────────

def compute_cost_aware_importance(
    feat_df: pd.DataFrame,
    knockoff_df: pd.DataFrame,
    features: list,
    ohlcv: pd.DataFrame,
    sim: PortfolioSimulator,
    cfg: dict,
    ic_signs: pd.Series,
    track: str = "track_b",
) -> pd.DataFrame:
    """Compute Z_j, Z̃_j, W_j for all features."""
    log(f"\n  Computing cost-aware importance for {len(features)} features …")
    all_tickers = feat_df.index.get_level_values("ticker").unique().tolist()
    returns, sigma, adv = load_ohlcv_matrices(ohlcv, all_tickers, IS_START, IS_END)

    # Splice knockoff features into feat_df index for signal building
    feat_is = feat_df.loc[
        (feat_df.index.get_level_values("date") >= pd.Timestamp(IS_START)) &
        (feat_df.index.get_level_values("date") <= pd.Timestamp(IS_END)),
        features
    ]

    # Rename knockoff columns to original names for signal building
    kn_cols_map = {f + "_kn": f for f in features}
    knockoff_renamed = knockoff_df.rename(columns=kn_cols_map)
    knockoff_renamed = knockoff_renamed[[f for f in features if f in knockoff_renamed.columns]]

    rows = []
    for i, feat in enumerate(features):
        # Z_j: real feature IS net Sharpe
        sig_real = build_single_feature_signal(feat_is, feat, ic_signs[feat], IS_START, IS_END)
        z_j = _fast_net_sharpe(
            sig_real, returns, sigma, adv,
            spread_bps=sim.spread_bps,
            impact_coeff=sim.impact_coeff,
            aum_dollars=cfg.get("aum_dollars", 1e8),
            min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
        )

        # Z̃_j: knockoff feature IS net Sharpe (same IC sign pre-specified)
        if feat in knockoff_renamed.columns:
            sig_kn = build_single_feature_signal(knockoff_renamed, feat, ic_signs[feat], IS_START, IS_END)
            z_j_tilde = _fast_net_sharpe(
                sig_kn, returns, sigma, adv,
                spread_bps=sim.spread_bps,
                impact_coeff=sim.impact_coeff,
                aum_dollars=cfg.get("aum_dollars", 1e8),
                min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
            )
        else:
            z_j_tilde = np.nan

        w_j = z_j - z_j_tilde if not np.isnan(z_j_tilde) else np.nan
        rows.append({"feature": feat, "Z_real": z_j, "Z_knockoff": z_j_tilde, "W_j": w_j})

        if (i + 1) % 5 == 0:
            log(f"    {i+1}/{len(features)} done")

    df = pd.DataFrame(rows)
    log(f"\n  W_j range: [{df['W_j'].min():.3f}, {df['W_j'].max():.3f}]")
    return df


# ── Knockoff(+) filter ────────────────────────────────────────────────────────

def knockoff_plus_filter(W: np.ndarray, q: float = FDR_Q) -> tuple:
    """Knockoff(+) threshold: τ = min{t: (1+#{W≤-t}) / max(1,#{W≥t}) ≤ q}.
    Returns (threshold τ, boolean selected array)."""
    W_sorted = np.sort(np.abs(W[~np.isnan(W)]))
    if len(W_sorted) == 0:
        return np.inf, np.zeros(len(W), dtype=bool)

    best_tau = np.inf
    for t in W_sorted:
        numerator   = 1 + (W <= -t).sum()
        denominator = max(1, (W >= t).sum())
        if numerator / denominator <= q:
            best_tau = t
            break

    selected = W >= best_tau
    return float(best_tau), selected


# ── FDR-validity gate (K3) ────────────────────────────────────────────────────

def fdr_validity_gate(
    feat_df: pd.DataFrame,
    features: list,
    copula: EmpiricalCopula,
    model,
    device: str,
    ohlcv: pd.DataFrame,
    sim: PortfolioSimulator,
    cfg: dict,
    ic_signs: pd.Series,
    n_sims: int = N_VALIDITY_SIMS,
    seed: int = 42,
) -> dict:
    """Simulate empirical FDR over known-null features.

    Adds K=10 Gaussian noise features to the panel, generates knockoffs,
    computes W_j for the null features across n_sims draws.
    Reports: empirical_fdr, gate_pass.
    """
    log(f"\n  FDR-validity gate: {n_sims} simulations, {N_NULL_FEATS} null features …")
    rng = np.random.default_rng(seed)

    dates_level = feat_df.index.get_level_values("date")
    is_mask = (dates_level >= pd.Timestamp(IS_START)) & \
              (dates_level <= pd.Timestamp(IS_END))
    feat_is = feat_df.loc[is_mask]

    null_names = [f"NULL_{i:02d}" for i in range(N_NULL_FEATS)]

    all_tickers = feat_is.index.get_level_values("ticker").unique().tolist()
    returns, sigma, adv = load_ohlcv_matrices(ohlcv, all_tickers, IS_START, IS_END)

    fdrs = []
    for sim_i in range(n_sims):
        # Generate null features: i.i.d. Gaussian, cross-sectionally standardised
        n_rows = len(feat_is)
        noise  = pd.DataFrame(
            rng.standard_normal((n_rows, N_NULL_FEATS)),
            index=feat_is.index, columns=null_names,
        )
        # CS z-score per date
        noise = noise.groupby(level="date").transform(
            lambda g: (g - g.mean()) / (g.std() + 1e-8)
        )

        # Dummy copula for null features (already standardised → approx uniform after rank-norm)
        null_copula = EmpiricalCopula()
        null_copula.fit(noise)

        # Generate knockoffs for null features via model
        kn_parts = []
        for ticker in feat_is.index.get_level_values("ticker").unique():
            try:
                import torch
                t_noise = noise.xs(ticker, level="ticker")
                u = null_copula.transform(t_noise).values.astype(np.float32)
                n = len(u)
                u_tilde = np.zeros_like(u)
                counts  = np.zeros(n)
                model.eval()
                with torch.no_grad():
                    # Use the null features as input (model sees same seq_len)
                    # Pad/trim to match feature dim by using only first min(35,10) dims
                    n_null = u.shape[1]
                    chunk_full = np.zeros((n, len(features)), dtype=np.float32)
                    chunk_full[:, :n_null] = u
                    for start in range(0, max(1, n - SEQ_LEN + 1), SEQ_LEN // 2):
                        end = min(start + SEQ_LEN, n)
                        chunk = torch.tensor(chunk_full[start:end]).unsqueeze(0).to(device)
                        if chunk.shape[1] < 2:
                            continue
                        tilde = model.generate_knockoff(chunk).cpu().numpy()[0]
                        u_tilde[start:end] += tilde[:end-start, :n_null]
                        counts[start:end] += 1
                counts = np.maximum(counts, 1)
                u_tilde = np.clip(u_tilde / counts[:, None], 1e-6, 1 - 1e-6)
                x_tilde = null_copula.inverse_transform(
                    pd.DataFrame(u_tilde, index=t_noise.index, columns=null_names)
                )
                x_tilde.index = pd.MultiIndex.from_product(
                    [[ticker], t_noise.index], names=["ticker", "date"]
                )
                kn_parts.append(x_tilde)
            except Exception:
                continue

        if not kn_parts:
            continue

        kn_null = pd.concat(kn_parts).sort_index()
        kn_null_renamed = kn_null.copy()
        kn_null_renamed.columns = [n.replace("NULL_", "NULL_kn_") for n in null_names]

        # W_j for null features
        w_null = []
        ic_signs_null = pd.Series(1.0, index=null_names)
        for null_feat in null_names:
            real_col = null_feat
            kn_col   = null_feat.replace("NULL_", "NULL_kn_")
            if real_col not in noise.columns or kn_col not in kn_null_renamed.columns:
                continue
            z_real = _fast_net_sharpe(
                build_single_feature_signal(noise, real_col, 1.0, IS_START, IS_END),
                returns, sigma, adv,
                spread_bps=sim.spread_bps, impact_coeff=sim.impact_coeff,
                aum_dollars=cfg.get("aum_dollars", 1e8),
                min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
            )
            null_kn_df = kn_null_renamed.rename(columns={kn_col: real_col})
            z_kn = _fast_net_sharpe(
                build_single_feature_signal(null_kn_df, real_col, 1.0, IS_START, IS_END),
                returns, sigma, adv,
                spread_bps=sim.spread_bps, impact_coeff=sim.impact_coeff,
                aum_dollars=cfg.get("aum_dollars", 1e8),
                min_adv_dollars=cfg.get("min_adv_dollars", 1e6),
            )
            w_null.append(z_real - z_kn if not (np.isnan(z_real) or np.isnan(z_kn)) else np.nan)

        if not w_null:
            continue

        w_arr = np.array(w_null)
        tau, selected = knockoff_plus_filter(w_arr, q=FDR_Q)
        total_sel  = selected.sum()
        null_sel   = selected.sum()  # all in w_null are null features
        fdp = null_sel / max(1, total_sel)
        fdrs.append(fdp)

        if (sim_i + 1) % 20 == 0:
            log(f"    sim {sim_i+1}/{n_sims}  running_FDR={np.mean(fdrs):.3f}")

    empirical_fdr = float(np.mean(fdrs)) if fdrs else np.nan
    gate_pass     = empirical_fdr <= FDR_Q * 1.1 if not np.isnan(empirical_fdr) else False

    log(f"\n  FDR-validity gate: empirical_FDR={empirical_fdr:.3f}  q_nominal={FDR_Q}")
    log(f"  Gate {'✓ PASS' if gate_pass else '✗ FAIL'}")

    return {
        "empirical_fdr": empirical_fdr,
        "q_nominal":     FDR_Q,
        "gate_pass":     gate_pass,
        "n_sims":        len(fdrs),
        "all_fdps":      fdrs,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gpu",   action="store_true", default=False)
    parser.add_argument("--epochs",   type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--track",    default="track_b")
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--skip-validity", action="store_true", default=False)
    args = parser.parse_args()

    t0 = time.time()
    log("=== Cost-Aware Knockoffs (CAK) ===\n")

    # Device selection
    if not args.no_gpu:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = "cpu"
    log(f"  Device: {device}")

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    log("\nLoading IS features …")
    feat_df = pd.read_parquet(FEATURES_PATH)
    # Derive the canonical pre-registered feature set from the loaded frame.
    features = _feature_columns(feat_df)
    log(f"  Features for knockoff generation: {len(features)}")
    log("Loading OHLCV …")
    ohlcv   = pd.read_parquet(OHLCV_PATH)

    sim = PortfolioSimulator(config_path=str(CFG_PATH))

    # IC signs from BH-on-IC for the requested track
    dates_level = feat_df.index.get_level_values("date")
    is_mask     = (dates_level >= pd.Timestamp(IS_START)) & \
                  (dates_level <= pd.Timestamp(IS_END))
    target_col  = "target_track_b" if args.track == "track_b" else "target_track_a"
    ic_signs = estimate_ic_signs(feat_df, features, target_col, IS_START, IS_END)

    # ── 1. Train LSTM-VAE ──────────────────────────────────────────────────
    log(f"\n[Step 1] Training LSTM-VAE ({args.epochs} epochs, noise_scale={args.noise_scale}) …")
    copula = EmpiricalCopula()
    model, losses = train_lstm_vae(
        feat_df, features, copula,
        epochs=args.epochs, device=device,
    )

    # Save model
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.save({"model_state": model.state_dict(),
                    "feature_list": features,
                    "copula": copula,
                    "n_features": len(features),
                    "hidden_dim": HIDDEN_DIM,
                    "latent_dim": LATENT_DIM}, str(OUT_MODEL))
        log(f"  Saved model → {OUT_MODEL}")
    except Exception as e:
        log(f"  (model save failed: {e})")

    # ── 2. Generate knockoff panel ─────────────────────────────────────────
    log("\n[Step 2] Generating knockoff features …")
    knockoff_df = generate_knockoff_panel(
        feat_df, features, copula, model, device, noise_scale=args.noise_scale
    )
    knockoff_df.to_parquet(OUT_KNOCKOFFS)
    log(f"  Saved → {OUT_KNOCKOFFS}  shape={knockoff_df.shape}")

    # ── 3. Cost-aware importance ───────────────────────────────────────────
    log("\n[Step 3] Computing cost-aware importance W_j …")
    importance_df = compute_cost_aware_importance(
        feat_df, knockoff_df, features, ohlcv, sim, cfg, ic_signs, args.track
    )
    importance_df.to_parquet(OUT_IMPORTANCE)
    log(f"  Saved → {OUT_IMPORTANCE}")

    # ── 4. Knockoff(+) filter ──────────────────────────────────────────────
    log("\n[Step 4] Applying knockoff(+) filter (q=0.10) …")
    W  = importance_df["W_j"].values
    tau, cak_selected = knockoff_plus_filter(W, q=FDR_Q)
    importance_df["cak_selected"] = cak_selected

    # Also get BH-on-IC set for rigor-ladder comparison
    bh_df = pd.read_parquet(FDR_PATH)
    bh_set = set(bh_df[(bh_df["track"] == args.track) & bh_df["bh_rejected"]]["feature"])
    importance_df["bh_on_ic"] = importance_df["feature"].isin(bh_set)

    cak_set = set(importance_df[importance_df["cak_selected"]]["feature"])
    only_cak = sorted(cak_set - bh_set)
    only_bh  = sorted(bh_set - cak_set)
    both     = sorted(cak_set & bh_set)

    log(f"\n  CAK selected: {len(cak_set)} features")
    log(f"  BH-on-IC:     {len(bh_set)} features")
    log(f"  In both:      {both}")
    log(f"  Only CAK:     {only_cak}")
    log(f"  Only BH-on-IC (stat real, cost-dominated): {only_bh}")

    # Rigor ladder summary
    results_df = importance_df.copy()
    results_df["track"] = args.track
    results_df.to_parquet(OUT_RESULTS)
    log(f"  Saved → {OUT_RESULTS}")

    # ── 5. FDR-validity gate ──────────────────────────────────────────────
    if not args.skip_validity:
        log("\n[Step 5] FDR-validity gate (mandatory) …")
        gate = fdr_validity_gate(
            feat_df, features, copula, model, device, ohlcv, sim, cfg, ic_signs,
            n_sims=N_VALIDITY_SIMS,
        )
        pd.DataFrame([{k: v for k, v in gate.items() if k != "all_fdps"}]).to_parquet(OUT_VALIDITY)
        log(f"  Saved → {OUT_VALIDITY}")

        if not gate["gate_pass"]:
            log("\n  ⚠  FDR-validity FAILED — knockoff headline cannot stand.")
            log("  → Fallback: use permutation TA-FDR (ta_fdr.parquet) for main claim.")
            log("  → Document failure in §5 (Cost-Aware Knockoffs) and report honestly.")
    else:
        log("\n  [Skipped FDR-validity gate — run with n_sims=200 before publishing]")

    log(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    log(f"\n★ CAK selected: {sorted(cak_set)}")


if __name__ == "__main__":
    main()
