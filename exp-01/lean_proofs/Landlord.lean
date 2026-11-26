import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases
import SlotReduction

/-!
# Landlord repair: the residual-deficit score (equation (25))

`A : Fin 4 → Fin N → ℝ` has row 0 equal to the cardinality row (all ones), rows 1..3
the resource rows.  `def_r` is the current overflow of row `r`, and `D_f` is the
overflow that remains if service `f` is evicted.
-/
namespace RandomizedDRL

/-- Current overflow of row `r`. -/
noncomputable def overflow (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ)
    (b : Fin 4 → ℝ) (r : Fin 4) : ℝ :=
  max 0 ((∑ n, A r n * x n) - b r)

/-- Residual-deficit score `D_f`, equation (25). -/
noncomputable def residualDeficit (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ)
    (b : Fin 4 → ℝ) (f : Fin N) : ℝ :=
  (∑ r : Fin 3, max 0 (overflow A x b r.succ - A r.succ f)) +
    max 0 (overflow A x b 0 - 1)

lemma overflow_nonneg (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ)
    (b : Fin 4 → ℝ) (r : Fin 4) : 0 ≤ overflow A x b r := by
  unfold overflow
  exact le_max_left _ _

lemma residualDeficit_nonneg (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ)
    (b : Fin 4 → ℝ) (f : Fin N) : 0 ≤ residualDeficit A x b f := by
  unfold residualDeficit
  apply add_nonneg
  · apply Finset.sum_nonneg
    intro r hr
    exact le_max_left _ _
  · exact le_max_left _ _

lemma max_sub_of_nonneg (u a : ℝ) (ha : 0 ≤ a) :
    max 0 (max 0 u - a) = max 0 (u - a) := by
  by_cases h : u ≤ 0
  · simp [h, ha]
    linarith
  · have h0 : 0 ≤ u := le_of_lt (lt_of_not_ge h)
    simp [h0]

/-- The score is exactly the sum of the row overflows after removing `f`, provided
`f` is present and the cardinality row has coefficient one.  This is the algebraic
content of equation (25). -/
lemma residualDeficit_eq_after_eviction
    (A : Fin 4 → Fin N → ℝ) (x : Fin N → ℝ) (b : Fin 4 → ℝ) (f : Fin N)
    (hf : x f = 1) (hA0 : ∀ n, A 0 n = 1) (hA : ∀ r n, 0 ≤ A r n) :
    residualDeficit A x b f =
      (∑ r : Fin 3, max 0 ((∑ n, A r.succ n * (if n = f then 0 else x n)) - b r.succ)) +
        max 0 ((∑ n, A 0 n * (if n = f then 0 else x n)) - b 0) := by
  classical
  have hsum (r : Fin 4) :
      ∑ n, A r n * (if n = f then 0 else x n) = (∑ n, A r n * x n) - A r f := by
    calc
      ∑ n, A r n * (if n = f then 0 else x n)
          = 0 + ∑ n ∈ (Finset.univ \ {f}), A r n * (if n = f then 0 else x n) := by
            rw [Finset.sum_eq_add_sum_diff_singleton (Finset.mem_univ f)]
            simp [hf]
      _ = ∑ n ∈ (Finset.univ \ {f}), A r n * x n := by
            simp only [zero_add]
            apply Finset.sum_congr rfl
            intro n hn
            have hnf : n ≠ f := by
              intro heq
              have : n ∈ (Finset.univ \ {f}) := hn
              simp [heq] at this
            simp [hnf]
      _ = (∑ n, A r n * x n) - A r f := by
            rw [Finset.sum_eq_add_sum_diff_singleton (Finset.mem_univ f)
              (fun n => A r n * x n)]
            simp [hf]
  unfold residualDeficit overflow
  simp_rw [hsum]
  have hmax (r : Fin 4) : max 0 (max 0 ((∑ n, A r n * x n) - b r) - A r f) =
      max 0 ((∑ n, A r n * x n) - A r f - b r) := by
    convert max_sub_of_nonneg (u := (∑ n, A r n * x n) - b r) (a := A r f) (hA r f) using 1 <;> ring
  simp_rw [hmax]
  simpa [hA0] using hmax 0

end RandomizedDRL
