import CavalFormal.Costs
import CavalFormal.ARV
import CavalFormal.FDR
import CavalFormal.TAFdr
import CavalFormal.Conformal

-- Every theorem should depend ONLY on [propext, Classical.choice, Quot.sound].
-- Any occurrence of `sorryAx` means the proof is fake.
#print axioms cSpread_nonneg
#print axioms cImpact_nonneg
#print axioms net_le_gross
#print axioms impactAvg_mono
#print axioms arvCard_antitone
#print axioms arv_antitone
#print axioms bhy_thr_le_bh_thr
#print axioms bhy_subset_bh
#print axioms net_stat_le_gross
#print axioms pval_antitone
#print axioms tafdr_reject_subset
#print axioms tafdr_conservative
#print axioms card_filter_perm
#print axioms card_filter_lt
#print axioms k_le_N
#print axioms split_conformal_coverage
