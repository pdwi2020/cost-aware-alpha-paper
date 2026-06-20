import Mathlib

variable {α : Type*} (C : Finset α) (netSR : α → ℝ)

/-- Number of configs with net Sharpe ratio strictly above threshold `tau`. -/
noncomputable def ARVcard (tau : ℝ) : ℕ := (C.filter (fun c => tau < netSR c)).card

/-- Alpha Robustness Volume: fraction of configs beating threshold `tau`. -/
noncomputable def ARV (tau : ℝ) : ℝ := (ARVcard C netSR tau : ℝ) / C.card

/-- The number of configurations above threshold is antitone in the threshold. -/
theorem arvCard_antitone {tau₁ tau₂ : ℝ} (h : tau₁ ≤ tau₂) :
    ARVcard C netSR tau₂ ≤ ARVcard C netSR tau₁ := by
  unfold ARVcard
  apply Finset.card_le_card
  intro x hx
  simp [Finset.mem_filter] at hx ⊢
  exact ⟨hx.1, lt_of_le_of_lt h hx.2⟩

/-- Alpha Robustness Volume is antitone in the threshold. -/
theorem arv_antitone {tau₁ tau₂ : ℝ} (h : tau₁ ≤ tau₂) :
    ARV C netSR tau₂ ≤ ARV C netSR tau₁ := by
  unfold ARV
  apply div_le_div_of_nonneg_right
  · exact_mod_cast arvCard_antitone C netSR h
  · exact Nat.cast_nonneg _
