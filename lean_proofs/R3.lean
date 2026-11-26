import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

/-!
# Randomized ski-rental threshold (R3)

The density is `f(θ) = e^{θ/M} / (M (e-1))` on `[0,M]`, with CDF
`F(θ) = (e^{θ/M}-1)/(e-1)`.  Inverse-transform sampling uses
`θ = M log(1 + u (e-1))`, `u ~ Uniform(0,1)`.
-/
namespace RandomizedDRL

noncomputable def r3CDF (M θ : ℝ) : ℝ :=
  (Real.exp (θ / M) - 1) / (Real.exp 1 - 1)

noncomputable def r3Inverse (M u : ℝ) : ℝ :=
  M * Real.log (1 + u * (Real.exp 1 - 1))

lemma exp_one_gt_one : 1 < Real.exp 1 := by
  exact Real.one_lt_exp_iff.mpr zero_lt_one

lemma r3Inverse_nonneg (M u : ℝ) (hM : 0 ≤ M) (hu : 0 ≤ u) :
    0 ≤ r3Inverse M u := by
  unfold r3Inverse
  apply mul_nonneg hM
  apply Real.log_nonneg
  have hE : 0 ≤ Real.exp 1 - 1 := by linarith [exp_one_gt_one]
  nlinarith [hu, hE]

lemma r3Inverse_le (M u : ℝ) (hM : 0 ≤ M) (hu0 : 0 ≤ u) (hu : u ≤ 1) :
    r3Inverse M u ≤ M := by
  unfold r3Inverse
  have hE : 0 ≤ Real.exp 1 - 1 := by linarith [exp_one_gt_one]
  have hlog : Real.log (1 + u * (Real.exp 1 - 1)) ≤ 1 := by
    calc
      Real.log (1 + u * (Real.exp 1 - 1)) ≤ Real.log (Real.exp 1) := by
        apply Real.log_le_log (by nlinarith [hu0, exp_one_gt_one])
        nlinarith [exp_one_gt_one, hu]
      _ = 1 := Real.log_exp 1
  nlinarith [hM, hlog]

lemma r3CDF_inverse (M u : ℝ) (hM : 0 < M) (hu : 0 ≤ u) :
    r3CDF M (r3Inverse M u) = u := by
  have hE : 0 ≤ Real.exp 1 - 1 := by linarith [exp_one_gt_one]
  have hpos : 0 < 1 + u * (Real.exp 1 - 1) := by nlinarith [hu, hE]
  unfold r3CDF r3Inverse
  have hdiv : M * Real.log (1 + u * (Real.exp 1 - 1)) / M =
      Real.log (1 + u * (Real.exp 1 - 1)) := by
    field_simp
  rw [hdiv, Real.exp_log hpos]
  have hden : Real.exp 1 - 1 ≠ 0 := by
    have := exp_one_gt_one
    linarith
  field_simp

lemma r3CDF_zero (M : ℝ) (hM : 0 < M) : r3CDF M 0 = 0 := by
  unfold r3CDF
  simp [hM.ne']

lemma r3CDF_one (M : ℝ) (hM : 0 < M) : r3CDF M M = 1 := by
  unfold r3CDF
  have hden : Real.exp 1 - 1 ≠ 0 := by
    have := exp_one_gt_one
    linarith
  field_simp

end RandomizedDRL
