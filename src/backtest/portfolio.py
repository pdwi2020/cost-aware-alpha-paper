"""Portfolio simulation: signal → position → execution → P&L.

Signal convention
-----------------
signal at date T → position effective at T (or T+1 with 1-day lag)
P&L at T = position_{T-1} × return_T  (return = close[T]/close[T-1] - 1)
Cost at T = |position_T - position_{T-1}| × total_cost_per_unit

All positions are in portfolio-weight space (fractions of gross AUM).
Long book sums to ~+1, short book to ~-1 → gross exposure ≈ 2 × (1x leveraged).
Scaled to gross_exposure_target from config (default 1.0 → 0.5x long, 0.5x short).
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent.parent

from src.backtest.almgren_chriss import (
    compute_spread_cost,
    compute_impact_cost,
    compute_borrow_cost,
)


# ---------------------------------------------------------------------------
# Screen 0 (look-ahead-free) helper — see src/data/screen0.py
# ---------------------------------------------------------------------------

def apply_s0_eligible(positions: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    """Apply look-ahead-free Screen 0 using the pre-computed s0_eligible flag.

    Screen 0 (look-ahead-free) — see src/data/screen0.py
    s0_eligible at date t is derived from price[t-1], trailing ADV[t-window:t-1],
    and PIT index membership — all known before the open at t.  No trade-date
    price is used, so there is zero look-ahead.

    Parameters
    ----------
    positions : date × ticker DataFrame of position weights (pre-renorm)
    feat_df   : MultiIndex (ticker, date) or (date, ticker) features DataFrame
                that contains an 's0_eligible' boolean column

    Returns
    -------
    positions : same shape, with ineligible names zeroed and L1 gross
                renormalised among the ELIGIBLE names (deployed capital unchanged).
                If 's0_eligible' is absent from feat_df, returns positions unchanged
                with a warning (so scripts degrade gracefully).
    """
    if "s0_eligible" not in feat_df.columns:
        import warnings
        warnings.warn(
            "[Screen0] s0_eligible column not found in features parquet; "
            "rebuild with build_features.py to enable look-ahead-free Screen 0. "
            "Proceeding WITHOUT Screen 0 filter.",
            stacklevel=2,
        )
        return positions

    elig_col = feat_df["s0_eligible"]
    # feat_df may already have a MultiIndex; unstack ticker → date×ticker boolean matrix
    elig_wide = elig_col.unstack(level="ticker")   # date × ticker
    elig_wide = elig_wide.reindex(index=positions.index, columns=positions.columns)
    elig_wide = elig_wide.fillna(False).astype(bool)

    positions = positions.where(elig_wide, 0.0)
    # Renormalise L1 gross among eligible names so capital stays deployed
    l1 = positions.abs().sum(axis=1).replace(0.0, np.nan)
    return positions.div(l1, axis=0).fillna(0.0)


class PortfolioSimulator:
    """Simulate strategy P&L with Almgren-Chriss execution costs."""

    def __init__(
        self,
        config_path: str = None,
        spread_bps: Optional[float] = None,
        impact_coeff: Optional[float] = None,
        borrow_tier: str = "easy",
    ):
        cfg_path = config_path or str(ROOT / "configs" / "backtest.yaml")
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self.spread_bps   = spread_bps   if spread_bps   is not None else self.cfg["spread_bps"]
        self.impact_coeff = impact_coeff if impact_coeff is not None else self.cfg["impact_coeff"]
        self.borrow_tier  = borrow_tier
        self.max_pos      = self.cfg["max_single_position"]
        self.gross_target = self.cfg["gross_exposure_target"]

    # ------------------------------------------------------------------
    # Position cap helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _waterfill_cap(
        w: pd.Series,
        max_pos: float,
        gross_target: float,
        tol: float = 1e-9,
        max_iter: int = 100,
    ) -> pd.Series:
        """Iterative water-fill cap: enforce |w_i| <= max_pos while keeping
        sum|w_i| == gross_target (when feasible).

        Algorithm
        ---------
        1. Normalise input so gross == gross_target (caller guarantees this,
           but we guard for safety).
        2. Identify names with |w_i| > max_pos; clamp them (frozen).
        3. Redistribute the gross freed by clamping proportionally among the
           unfrozen names (preserving relative sizes and signs).
        4. Repeat until no unfrozen name exceeds max_pos (convergence).

        Feasibility: if n_active * max_pos < gross_target the cap constraint
        cannot be satisfied while meeting the gross target.  In that case every
        name is capped at max_pos and the returned gross is n_active * max_pos
        (strictly less than gross_target).  The position cap is NEVER breached.
        """
        if max_pos <= 0 or len(w) == 0:
            return w.copy()

        w = w.copy()

        # Guard: re-normalise to gross_target if needed
        gross = w.abs().sum()
        if gross > tol:
            w = w / gross * gross_target

        frozen = pd.Series(False, index=w.index)

        for _ in range(max_iter):
            over = w.abs() > max_pos + tol
            newly_frozen = over & ~frozen
            if not newly_frozen.any():
                break  # converged

            # Clamp newly-frozen names
            w[over] = w[over].clip(-max_pos, max_pos)
            frozen = frozen | over

            unfrozen = ~frozen
            if not unfrozen.any():
                # All names frozen — feasibility exhausted
                break

            # How much gross do the unfrozen names currently carry?
            gross_frozen   = w[frozen].abs().sum()
            gross_unfrozen = w[unfrozen].abs().sum()
            gross_needed   = gross_target - gross_frozen

            # Feasibility check: can unfrozen names absorb gross_needed?
            n_unfrozen = unfrozen.sum()
            max_unfrozen_gross = n_unfrozen * max_pos
            if gross_needed > max_unfrozen_gross + tol:
                # Infeasible — cap every unfrozen name at max_pos too
                w[unfrozen] = w[unfrozen].clip(-max_pos, max_pos)
                frozen = frozen | unfrozen
                break

            # Scale unfrozen proportionally to absorb the remaining gross
            if gross_unfrozen > tol:
                w[unfrozen] = w[unfrozen] / gross_unfrozen * gross_needed
            # (if gross_unfrozen == 0 there is nothing to redistribute)

        return w

    # ------------------------------------------------------------------
    # Signal → Positions
    # ------------------------------------------------------------------

    def signal_to_positions(
        self,
        signals: pd.DataFrame,
        lag: int = 1,
        rebal_freq: int = 1,
    ) -> pd.DataFrame:
        """Convert signal scores to target portfolio weights.

        Cross-sectionally z-scores the signal per date, normalises to
        gross_exposure_target, and caps individual positions at max_single_position.

        Args:
            signals:    (date × ticker) signal scores. Higher = long.
            lag:        Implementation lag in days (1 = trade next day open).
            rebal_freq: Rebalancing frequency in days. 1 = daily (default),
                        5 = weekly, 21 = monthly. Between rebalance dates the
                        previous position is held unchanged.

        Returns:
            (date × ticker) portfolio weights (signed fractions of AUM).
        """
        dates   = signals.index
        weights = pd.DataFrame(0.0, index=dates, columns=signals.columns)

        last_w  = pd.Series(0.0, index=signals.columns)
        day_ctr = 0

        for d in dates:
            if rebal_freq > 1 and day_ctr % rebal_freq != 0:
                # Hold previous position
                weights.loc[d, last_w.index] = last_w
                day_ctr += 1
                continue

            row = signals.loc[d].dropna()
            if len(row) < 10:
                weights.loc[d, last_w.index] = last_w
                day_ctr += 1
                continue

            mu  = row.mean()
            sig = row.std()
            if sig < 1e-8:
                weights.loc[d, last_w.index] = last_w
                day_ctr += 1
                continue

            z = (row - mu) / sig        # cross-sectional z-score

            # Normalise so gross ≈ gross_target
            gross = z.abs().sum()
            if gross < 1e-8:
                weights.loc[d, last_w.index] = last_w
                day_ctr += 1
                continue
            w = z / gross * self.gross_target

            # Cap individual positions (iterative water-fill)
            w = self._waterfill_cap(w, self.max_pos, self.gross_target)

            weights.loc[d, w.index] = w
            last_w = weights.loc[d].copy()
            day_ctr += 1

        # Implementation lag: shift positions forward
        if lag > 0:
            weights = weights.shift(lag)

        return weights.fillna(0.0)

    # ------------------------------------------------------------------
    # P&L simulation
    # ------------------------------------------------------------------

    def simulate_pnl(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        vol: Optional[pd.DataFrame] = None,
        adv_dollars: Optional[pd.DataFrame] = None,
        aum_dollars: float = 1e8,
        min_adv_dollars: float = 1e6,
    ) -> pd.DataFrame:
        """Compute daily gross and net P&L.

        Args:
            positions:       (date × ticker) portfolio weights.
            returns:         (date × ticker) next-day close-to-close returns.
            vol:             (date × ticker) daily volatility (for impact cost).
            adv_dollars:     (date × ticker) average daily dollar volume.
            aum_dollars:     Assumed portfolio AUM in dollars (converts weight
                             changes to dollar notional for AC impact model).
            min_adv_dollars: Minimum ADV filter. Stocks with ADV below this
                             threshold are excluded from positions (set to 0)
                             to prevent bankrupt/delisted stocks from generating
                             unrealistic market-impact costs. Default $1M.

        Returns:
            DataFrame with columns: gross_pnl, spread_cost, impact_cost,
            borrow_cost, total_cost, net_pnl, turnover.
        """
        common_dates    = positions.index.intersection(returns.index)
        common_tickers  = positions.columns.intersection(returns.columns)
        pos  = positions.loc[common_dates, common_tickers]
        ret  = returns.loc[common_dates, common_tickers]

        delta_pos = pos.diff().fillna(pos)     # position changes (day 0: full position)

        # Fallback vol and ADV if not provided
        if vol is None:
            vol = pd.DataFrame(0.02, index=pos.index, columns=pos.columns)
        else:
            vol = vol.reindex(index=pos.index, columns=pos.columns).fillna(0.02)

        if adv_dollars is None:
            adv_dollars = pd.DataFrame(1e8, index=pos.index, columns=pos.columns)
        else:
            adv_dollars = adv_dollars.reindex(
                index=pos.index, columns=pos.columns
            ).fillna(1e8)

        records = []
        for d in common_dates:
            p_d     = pos.loc[d]
            r_d     = ret.loc[d]
            dp_d    = delta_pos.loc[d].abs()    # |Δw| per ticker
            v_d     = vol.loc[d]
            adv_d   = adv_dollars.loc[d]

            # Zero out positions in stocks with insufficient liquidity.
            # Prevents bankrupt/delisted stocks (ADV → 0) from generating
            # unrealistic AC impact costs.
            liquid = adv_d >= min_adv_dollars
            p_d  = p_d.where(liquid, 0.0)
            dp_d = dp_d.where(liquid, 0.0)

            gross    = float((p_d * r_d).sum())
            turnover = float(dp_d.sum())

            # Spread: paid on each trade (half spread per side = full spread round-trip)
            sc = float(dp_d.sum()) * compute_spread_cost(self.spread_bps)

            # Market impact (Almgren-Chriss):
            #   trade_dollars_i = |Δw_i| × AUM
            #   participation_i = trade_dollars_i / ADV_i
            #   impact_frac_i   = η × vol_i × sqrt(participation_i)
            #   cost_i          = impact_frac_i × |Δw_i|   (as fraction of AUM)
            participation = (dp_d * aum_dollars) / adv_d.clip(lower=1.0)
            ic_per_stock  = self.impact_coeff * v_d * np.sqrt(participation)
            ic = float((ic_per_stock * dp_d).sum())

            # Borrow cost: only for short positions held overnight
            bc = float(
                p_d.clip(upper=0.0).abs().sum()
                * (self.cfg.get("borrow_cost_easy_bps", 0) / 10_000 / 252)
            )

            total_cost = sc + ic + bc
            net        = gross - total_cost

            records.append({
                "date":        d,
                "gross_pnl":   gross,
                "spread_cost": sc,
                "impact_cost": ic,
                "borrow_cost": bc,
                "total_cost":  total_cost,
                "net_pnl":     net,
                "turnover":    turnover,
            })

        return pd.DataFrame(records).set_index("date")

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, pnl_df: pd.DataFrame) -> Dict[str, float]:
        """Annualised Sharpe, max drawdown, hit rate, turnover from daily P&L."""
        metrics = {}
        for col in ["gross_pnl", "net_pnl"]:
            s    = pnl_df[col].dropna()
            mean = s.mean()
            std  = s.std()
            sr   = float(mean / std * np.sqrt(252)) if std > 1e-10 else np.nan
            cum  = (1 + s).cumprod()
            dd   = float((cum / cum.cummax() - 1).min())
            metrics[f"{col}_annual"]  = float(mean * 252)
            metrics[f"{col}_sharpe"]  = sr
            metrics[f"{col}_max_dd"]  = dd
            metrics[f"{col}_hit_rate"] = float((s > 0).mean())

        metrics["daily_turnover"]  = float(pnl_df["turnover"].mean())
        metrics["annual_turnover"] = float(pnl_df["turnover"].mean() * 252)
        metrics["cost_drag_bps"]   = float(pnl_df["total_cost"].mean() * 252 * 10_000)
        return metrics
