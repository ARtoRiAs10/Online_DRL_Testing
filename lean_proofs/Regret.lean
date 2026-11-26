import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases

/-!
# Proposition 1 ingredients: diameter and gradient bound

The full OGD regret inequality is a standard theorem about projected gradient descent;
the two geometric ingredients used by the proposition are formalized here.
-/
namespace RandomizedDRL

lemma sub_sq_le_one (p q : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1) : (p - q) ^ 2 ≤ 1 := by
  nlinarith [sq_nonneg (p - q), sq_nonneg (1 - (p - q)), sq_nonneg (1 + (p - q))]

lemma sum_sub_sq_le_card (p q : Fin N → ℝ)
    (hp0 : ∀ n, 0 ≤ p n) (hp1 : ∀ n, p n ≤ 1)
    (hq0 : ∀ n, 0 ≤ q n) (hq1 : ∀ n, q n ≤ 1) :
    ∑ n, (p n - q n) ^ 2 ≤ N := by
  calc
    ∑ n, (p n - q n) ^ 2 ≤ ∑ n, (1 : ℝ) := by
      apply Finset.sum_le_sum
      intro n hn
      exact sub_sq_le_one (p n) (q n) (hp0 n) (hp1 n) (hq0 n) (hq1 n)
    _ = N := by simp

/-- The diameter of the unit box in `ℝ^N` is at most `√N`. -/
lemma diameter_le_sqrt (p q : Fin N → ℝ)
    (hp0 : ∀ n, 0 ≤ p n) (hp1 : ∀ n, p n ≤ 1)
    (hq0 : ∀ n, 0 ≤ q n) (hq1 : ∀ n, q n ≤ 1) :
    Real.sqrt (∑ n, (p n - q n) ^ 2) ≤ Real.sqrt N := by
  apply Real.sqrt_le_sqrt
  exact sum_sub_sq_le_card p q hp0 hp1 hq0 hq1

/-- Instantaneous gradient, equation (18). -/
noncomputable def grad (nstar : Fin N) (q dm l : ℝ) : Fin N → ℝ :=
  fun n => if n = nstar then -((1 - q) * l + q * min l dm) else 0

lemma grad_norm_sq_le (nstar : Fin N) (q dm l lmax : ℝ)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hl0 : 0 ≤ l) (hlmax : l ≤ lmax) (hdm : 0 ≤ dm) :
    ∑ n, (grad nstar q dm l n) ^ 2 ≤ lmax ^ 2 := by
  classical
  rw [Finset.sum_eq_single nstar]
  · have hterm : 0 ≤ (1 - q) * l + q * min l dm := by
      apply add_nonneg
      · exact mul_nonneg (by linarith) hl0
      · exact mul_nonneg hq0 (le_min hl0 hdm)
    have hterm_le : (1 - q) * l + q * min l dm ≤ l := by
      have hmin : min l dm ≤ l := min_le_left _ _
      nlinarith
    simp [grad]
    nlinarith [hterm, hterm_le, hlmax]
  · intro n hn hns
    simp [grad, hns]
  · simp

end RandomizedDRL
