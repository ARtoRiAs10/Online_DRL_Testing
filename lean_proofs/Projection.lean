import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases

/-!
# Projection onto the multi-resource polytope (equations (20)-(24))

The constraints are normalized to `A_n · p ≤ 1` (row 0 is cardinality, rows 1..3 are
CPU/RAM/disk).  The KKT threshold operator is
`p_n(ρ) = clip(z_n - ∑_r ρ_r A_{n,r}, 0, 1)`.
-/
namespace RandomizedDRL

/-- Clipping to the unit interval, written with `if` to make the case analysis explicit. -/
noncomputable def clip01 (z : ℝ) : ℝ := if z ≤ 0 then 0 else if z ≤ 1 then z else 1

lemma clip01_eq_minmax (z : ℝ) : clip01 z = min 1 (max 0 z) := by
  by_cases h0 : z ≤ 0
  · simp [clip01, h0]
  · have h0' : 0 < z := lt_of_not_ge h0
    by_cases h1 : z ≤ 1
    · simp [clip01, h0, h1]
      exact le_of_lt h0'
    · have h1' : 1 < z := lt_of_not_ge h1
      simp [clip01, h0, h1]
      right; linarith

lemma clip01_nonneg (z : ℝ) : 0 ≤ clip01 z := by
  unfold clip01
  split_ifs with h0 h1
  · exact le_refl 0
  · exact le_of_lt (lt_of_not_ge h0)
  · exact zero_le_one

lemma clip01_le_one (z : ℝ) : clip01 z ≤ 1 := by
  unfold clip01
  split_ifs with h0 h1
  · exact zero_le_one
  · exact h1
  · exact le_refl 1

/-- The one-dimensional quadratic minimized by the clipping operator. -/
lemma quadratic_min (z w y : ℝ) (hy0 : 0 ≤ y) (hy1 : y ≤ 1) :
    1 / 2 * (clip01 (z - w) - z) ^ 2 + w * clip01 (z - w)
      ≤ 1 / 2 * (y - z) ^ 2 + w * y := by
  by_cases h0 : z - w ≤ 0
  · have hc : clip01 (z - w) = 0 := by simp [clip01, h0]
    rw [hc]
    nlinarith
  · by_cases h1 : z - w ≤ 1
    · have hc : clip01 (z - w) = z - w := by simp [clip01, h0, h1]
      rw [hc]
      have hsq := sq_nonneg (w + (y - z))
      nlinarith
    · have hc : clip01 (z - w) = 1 := by simp [clip01, h0, h1]
      rw [hc]
      nlinarith

/-- KKT threshold operator, equation (21), in normalized form. -/
noncomputable def threshold (z : Fin N → ℝ) (A : Fin N → Fin 4 → ℝ) (ρ : Fin 4 → ℝ) (n : Fin N) : ℝ :=
  clip01 (z n - ∑ r, ρ r * A n r)

/-- The threshold is the coordinate-wise minimizer of the box-constrained quadratic
in the Lagrangian (equation (22)). -/
lemma threshold_minimizes_coord (z : Fin N → ℝ) (A : Fin N → Fin 4 → ℝ)
    (ρ : Fin 4 → ℝ) (n : Fin N) (y : ℝ) (hy0 : 0 ≤ y) (hy1 : y ≤ 1) :
    1 / 2 * (threshold z A ρ n - z n) ^ 2 + (∑ r, ρ r * A n r) * threshold z A ρ n
      ≤ 1 / 2 * (y - z n) ^ 2 + (∑ r, ρ r * A n r) * y := by
  exact quadratic_min (z n) (∑ r, ρ r * A n r) y hy0 hy1

/-- The dual objective (equation (22)) at a *fixed* primal point.  The envelope
theorem identifies its ρ-gradient with the constraint residual; the fixed-p identity
below is the algebraic part of that statement. -/
noncomputable def lagrangian_fixed (z p : Fin N → ℝ) (A : Fin N → Fin 4 → ℝ)
    (b : Fin 4 → ℝ) (ρ : Fin 4 → ℝ) : ℝ :=
  1 / 2 * ∑ n, (p n - z n) ^ 2 + ∑ r, ρ r * ((∑ n, A n r * p n) - b r)

lemma lagrangian_fixed_shift (z p : Fin N → ℝ) (A : Fin N → Fin 4 → ℝ)
    (b : Fin 4 → ℝ) (ρ : Fin 4 → ℝ) (r : Fin 4) (t : ℝ) :
    lagrangian_fixed z p A b (fun s => ρ s + if s = r then t else 0) =
      lagrangian_fixed z p A b ρ + t * ((∑ n, A n r * p n) - b r) := by
  classical
  unfold lagrangian_fixed
  simp_rw [add_mul]
  rw [Finset.sum_add_distrib]
  have hif : (∑ s, (if s = r then t else 0) *
      ((∑ n, A n s * p n) - b s)) = t * ((∑ n, A n r * p n) - b r) := by
    rw [Finset.sum_eq_single r]
    · simp
    · intro s hs hsr
      simp [hsr]
    · simp
  rw [hif]
  ring

/-- The final homogeneous rescaling, equation (24), in normalized form.
If `γ ≥ 1` and `γ ≥ load_r` for every row, then `p / γ` is feasible. -/
lemma rescale_feasible (p : Fin N → ℝ) (A : Fin N → Fin 4 → ℝ)
    (hp0 : ∀ n, 0 ≤ p n) (hp1 : ∀ n, p n ≤ 1) (γ : ℝ)
    (hγ1 : 1 ≤ γ) (hγ : ∀ r, (∑ n, A n r * p n) ≤ γ) :
    (∀ n, 0 ≤ p n / γ) ∧ (∀ n, p n / γ ≤ 1) ∧
      (∀ r, ∑ n, A n r * (p n / γ) ≤ 1) := by
  have hγpos : 0 < γ := lt_of_lt_of_le zero_lt_one hγ1
  constructor
  · intro n; exact div_nonneg (hp0 n) (le_of_lt hγpos)
  constructor
  · intro n
    rw [div_le_one hγpos]
    exact le_trans (hp1 n) hγ1
  · intro r
    have hterm : ∀ n, A n r * (p n / γ) = (A n r * p n) / γ := by
      intro n; ring
    have hsum : ∑ n, A n r * p n / γ = (∑ n, A n r * p n) / γ := by
      calc
        ∑ n, A n r * p n / γ = (1 / γ) * ∑ n, A n r * p n := by
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro n hn
          ring
        _ = (∑ n, A n r * p n) / γ := by ring
    have hsum' : ∑ n, A n r * (p n / γ) = (∑ n, A n r * p n) / γ := by
      rw [Finset.sum_congr rfl (fun n _ => hterm n)]
      exact hsum
    rw [hsum']
    exact (div_le_one hγpos).2 (hγ r)

end RandomizedDRL
