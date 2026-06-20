import Mathlib

/-- BHY threshold `kq/(mH)` is at most the BH threshold `kq/m` when `H ≥ 1`. -/
theorem bhy_thr_le_bh_thr {k q m H : ℝ} (hk : 0 ≤ k) (hq : 0 ≤ q) (hm : 0 < m)
    (hH : 1 ≤ H) :
    k * q / (m * H) ≤ k * q / m := by
  apply div_le_div_of_nonneg_left (mul_nonneg hk hq) hm
  exact le_mul_of_one_le_right (le_of_lt hm) hH

/-- Every rejection passing the BHY threshold also passes the BH threshold. -/
theorem bhy_subset_bh {ι : Type*} (S : Finset ι) (p bhThr bhyThr : ι → ℝ)
    (h : ∀ i, bhyThr i ≤ bhThr i) :
    S.filter (fun i => p i ≤ bhyThr i) ⊆ S.filter (fun i => p i ≤ bhThr i) := by
  intro x hx
  simp [Finset.mem_filter] at *
  exact ⟨hx.1, le_trans hx.2 (h x)⟩
