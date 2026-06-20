import Mathlib

/-- Spread cost: half-spread `h` in basis points times one-way turnover `DeltaW`. -/
noncomputable def cSpread (DeltaW h : ℝ) : ℝ := DeltaW * (h / 10000)

/-- Almgren-Chriss market impact cost. -/
noncomputable def cImpact (eta sigma AUM ADV DeltaW : ℝ) : ℝ :=
  eta * sigma * Real.sqrt (DeltaW * AUM / ADV) * DeltaW

/-- Average impact cost per unit turnover, used for monotonicity. -/
noncomputable def impactAvg (eta sigma AUM ADV DeltaW : ℝ) : ℝ :=
  eta * sigma * Real.sqrt (DeltaW * AUM / ADV)

/-- Spread costs are nonnegative when turnover and half-spread are nonnegative. -/
theorem cSpread_nonneg {DeltaW h : ℝ} (hDeltaW : 0 ≤ DeltaW) (hh : 0 ≤ h) :
    0 ≤ cSpread DeltaW h := by
  unfold cSpread
  exact mul_nonneg hDeltaW (div_nonneg hh (by norm_num))

/-- Market impact costs are nonnegative under nonnegative inputs. -/
theorem cImpact_nonneg {eta sigma AUM ADV DeltaW : ℝ} (heta : 0 ≤ eta) (hsigma : 0 ≤ sigma)
    (hAUM : 0 ≤ AUM) (hADV : 0 ≤ ADV) (hDeltaW : 0 ≤ DeltaW) :
    0 ≤ cImpact eta sigma AUM ADV DeltaW := by
  unfold cImpact
  exact mul_nonneg (mul_nonneg (mul_nonneg heta hsigma) (Real.sqrt_nonneg _)) hDeltaW

/-- Net PnL is bounded above by gross PnL when all costs are nonnegative. -/
theorem net_le_gross (pnlGross cs ci cb : ℝ) (hcs : 0 ≤ cs) (hci : 0 ≤ ci)
    (hcb : 0 ≤ cb) :
    pnlGross - (cs + ci + cb) ≤ pnlGross := by
  linarith

/-- Average impact cost is monotone in turnover under nonnegative parameters. -/
theorem impactAvg_mono {eta sigma AUM ADV x y : ℝ} (heta : 0 ≤ eta) (hsigma : 0 ≤ sigma)
    (hAUM : 0 ≤ AUM) (hADV : 0 ≤ ADV) (hx : 0 ≤ x) (hxy : x ≤ y) :
    impactAvg eta sigma AUM ADV x ≤ impactAvg eta sigma AUM ADV y := by
  unfold impactAvg
  have hcoeff : 0 ≤ eta * sigma := mul_nonneg heta hsigma
  have hmul : x * AUM ≤ y * AUM := mul_le_mul_of_nonneg_right hxy hAUM
  have hdiv : x * AUM / ADV ≤ y * AUM / ADV := div_le_div_of_nonneg_right hmul hADV
  have hsqrt :
      Real.sqrt (x * AUM / ADV) ≤ Real.sqrt (y * AUM / ADV) :=
    Real.sqrt_le_sqrt hdiv
  exact mul_le_mul_of_nonneg_left hsqrt hcoeff
