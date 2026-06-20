import Mathlib

/-- Net test statistic is at most gross because cost is subtracted. -/
theorem net_stat_le_gross (Tgross cost : ℝ) (hc : 0 ≤ cost) :
    Tgross - cost ≤ Tgross := by
  linarith

variable {B : ℕ} (Tb : Fin B → ℝ)

/-- Number of permuted statistics at least as large as `T`. -/
noncomputable def exceed (T : ℝ) : ℕ := (Finset.univ.filter (fun b => T ≤ Tb b)).card

/-- Permutation p-value with one-count smoothing. -/
noncomputable def pval (T : ℝ) : ℝ := (1 + (exceed Tb T : ℝ)) / (1 + B)

/-- The smoothed permutation p-value is antitone in the observed statistic. -/
theorem pval_antitone {T₁ T₂ : ℝ} (h : T₁ ≤ T₂) : pval Tb T₂ ≤ pval Tb T₁ := by
  unfold pval exceed
  have hcard :
      (Finset.univ.filter (fun b => T₂ ≤ Tb b)).card ≤
        (Finset.univ.filter (fun b => T₁ ≤ Tb b)).card := by
    apply Finset.card_le_card
    intro b hb
    simp [Finset.mem_filter] at hb ⊢
    exact le_trans h hb
  apply div_le_div_of_nonneg_right
  · exact add_le_add_left (by exact_mod_cast hcard) 1
  · exact add_nonneg zero_le_one (Nat.cast_nonneg B)

/-- Net rejections are contained in gross rejections when gross p-values are no larger. -/
theorem tafdr_reject_subset {ι : Type*} (F : Finset ι) (pnet pgross : ι → ℝ) (t : ℝ)
    (h : ∀ i, pgross i ≤ pnet i) :
    F.filter (fun i => pnet i ≤ t) ⊆ F.filter (fun i => pgross i ≤ t) := by
  intro x hx
  simp [Finset.mem_filter] at *
  exact ⟨hx.1, le_trans (h x) hx.2⟩

/-- The number of net rejections is bounded by the number of gross rejections. -/
theorem tafdr_conservative {ι : Type*} (F : Finset ι) (pnet pgross : ι → ℝ) (t : ℝ)
    (h : ∀ i, pgross i ≤ pnet i) :
    (F.filter (fun i => pnet i ≤ t)).card ≤ (F.filter (fun i => pgross i ≤ t)).card := by
  apply Finset.card_le_card
  exact tafdr_reject_subset F pnet pgross t h
