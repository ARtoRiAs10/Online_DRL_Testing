import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases

/-!
# Latency model and resource metrics (equations (27)-(34))
-/
namespace RandomizedDRL

/-- Physical latency, equation (27).  `cached` is `x_{n,t}=1`, `downloading` is
`y_{n,t}=1`. -/
noncomputable def physicalLatency (l dm : ℝ) (cached downloading : Bool) : ℝ :=
  if cached then 0 else if downloading then min l dm else l

lemma physicalLatency_nonneg (l dm : ℝ) (hl : 0 ≤ l) (hdm : 0 ≤ dm)
    (cached downloading : Bool) : 0 ≤ physicalLatency l dm cached downloading := by
  unfold physicalLatency
  split_ifs
  · exact le_refl 0
  · exact le_min hl hdm
  · exact hl

lemma physicalLatency_le (l dm : ℝ) (hl : 0 ≤ l) (hdm : 0 ≤ dm)
    (cached downloading : Bool) : physicalLatency l dm cached downloading ≤ l := by
  unfold physicalLatency
  split_ifs
  · exact hl
  · exact min_le_left _ _
  · exact le_refl l

/-- Expected latency, equation (28). -/
noncomputable def expectedLatency (p q l dm : ℝ) : ℝ :=
  (1 - p) * ((1 - q) * l + q * min l dm)

lemma expectedLatency_eq_bernoulli (p q l dm : ℝ) :
    expectedLatency p q l dm = p * 0 + (1 - p) * ((1 - q) * l + q * min l dm) := by
  rw [expectedLatency]
  ring

/-- Surrogate latency, equation (29). -/
noncomputable def surrogate (lam p q l dm : Fin N → ℝ) : ℝ :=
  ∑ n, lam n * (1 - p n) * ((1 - q n) * l n + q n * min (l n) (dm n))

lemma surrogate_affine (lam p q l dm : Fin N → ℝ) :
    surrogate lam p q l dm =
      (∑ n, lam n * ((1 - q n) * l n + q n * min (l n) (dm n))) -
        ∑ n, lam n * p n * ((1 - q n) * l n + q n * min (l n) (dm n)) := by
  unfold surrogate
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro n hn
  ring

/-- Resource utilization, equation (30). -/
noncomputable def utilization (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ) (r : Fin 4) : ℝ :=
  ∑ n, A r n * x n

/-- Bottleneck utilization, equation (31). -/
noncomputable def bottleneckUtil (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ) : ℝ :=
  max (utilization A x 1) (max (utilization A x 2) (utilization A x 3))

lemma bottleneckUtil_ge (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ) (r : Fin 3) :
    utilization A x r.succ ≤ bottleneckUtil A x := by
  fin_cases r <;> simp [bottleneckUtil, le_max_iff]

/-- Imbalance, equation (32). -/
noncomputable def imbalance (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ) : ℝ :=
  bottleneckUtil A x -
    min (utilization A x 1) (min (utilization A x 2) (utilization A x 3))

lemma imbalance_nonneg (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ) :
    0 ≤ imbalance A x := by
  unfold imbalance
  exact sub_nonneg.mpr (le_trans (min_le_left _ _) (le_max_left _ _))

/-- Stranded capacity, equation (33). -/
noncomputable def stranded (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ) : ℝ :=
  ∑ r : Fin 3, (bottleneckUtil A x - utilization A x r.succ)

lemma stranded_nonneg (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ) :
    0 ≤ stranded A x := by
  unfold stranded
  apply Finset.sum_nonneg
  intro r hr
  exact sub_nonneg.mpr (bottleneckUtil_ge A x r)

end RandomizedDRL
