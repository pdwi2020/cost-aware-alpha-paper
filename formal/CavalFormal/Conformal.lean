import Mathlib

/-- Permuting `Fin N` does not change the number of indices whose permuted value is below `k`. -/
theorem card_filter_perm {N : ℕ} (rho : Equiv.Perm (Fin N)) (k : ℕ) :
    (Finset.univ.filter (fun i => (rho i : ℕ) < k)).card =
    (Finset.univ.filter (fun j : Fin N => (j : ℕ) < k)).card := by
  apply Finset.card_nbij' (fun i => rho i) (fun j => rho.symm j)
  · intro i hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
    exact hi
  · intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
    rwa [rho.apply_symm_apply j]
  · intro i hi
    exact rho.symm_apply_apply i
  · intro j hj
    exact rho.apply_symm_apply j

/-- In `Fin N`, exactly `k` indices have value below `k`, provided `k ≤ N`. -/
theorem card_filter_lt {N : ℕ} (k : ℕ) (hk : k ≤ N) :
    (Finset.univ.filter (fun j : Fin N => (j : ℕ) < k)).card = k := by
  have h1 : (Finset.univ.filter (fun j : Fin N => (j : ℕ) < k)).card =
            Fintype.card {j : Fin N // (j : ℕ) < k} := by
    rw [Fintype.card_subtype]
  rw [h1]
  exact Fintype.card_fin_lt_of_le hk

/-- The conformal ceiling index is at most the calibration size. -/
theorem k_le_N {N : ℕ} {alpha : ℝ} (halpha0 : 0 ≤ alpha) (hN : 0 < N) :
    ⌈(1 - alpha) * (N : ℝ)⌉₊ ≤ N := by
  rw [Nat.ceil_le]
  have hN_nonneg : (0 : ℝ) ≤ N := le_of_lt (Nat.cast_pos.mpr hN)
  have hfactor : 1 - alpha ≤ 1 := by linarith
  simpa using mul_le_mul_of_nonneg_right hfactor hN_nonneg

/-- Split conformal coverage lower bound under a permutation of calibration ranks. -/
theorem split_conformal_coverage {N : ℕ} (hN : 0 < N) (rho : Equiv.Perm (Fin N))
    {alpha : ℝ} (halpha0 : 0 ≤ alpha) (halpha1 : alpha ≤ 1) :
    (1 - alpha) ≤
      ((Finset.univ.filter
        (fun i => (rho i : ℕ) < ⌈(1 - alpha) * (N : ℝ)⌉₊)).card : ℝ) / N := by
  have _hleft_nonneg : 0 ≤ 1 - alpha := sub_nonneg.mpr halpha1
  have hNR : (0 : ℝ) < N := Nat.cast_pos.mpr hN
  rw [card_filter_perm rho ⌈(1 - alpha) * (N : ℝ)⌉₊]
  rw [card_filter_lt _ (k_le_N halpha0 hN)]
  rw [le_div_iff₀ hNR]
  exact Nat.le_ceil _
