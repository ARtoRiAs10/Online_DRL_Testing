import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases

/-!
# Slot reductions (equations (13)-(17) of the draft)

We formalize the finite-dimensional (three resource) algebra.  A demand vector is a
function `a : Fin 3 → ℝ`; a slot reduction is a map `phi` from demand vectors to
non-negative reals.  The scalar slot weight is `C_slot * phi a`, so the slot constraint
`∑ w_n p_n ≤ C_slot` is exactly `∑ phi(a_n) p_n ≤ 1`.
-/
namespace RandomizedDRL

/-- The maximum of three non-negative resource demands. -/
def max3 (a : Fin 3 → ℝ) : ℝ := max (a 0) (max (a 1) (a 2))

lemma max3_ge (a : Fin 3 → ℝ) (r : Fin 3) : a r ≤ max3 a := by
  fin_cases r <;> simp [max3, le_max_iff]

lemma max3_le {a : Fin 3 → ℝ} {c : ℝ}
    (h0 : a 0 ≤ c) (h1 : a 1 ≤ c) (h2 : a 2 ≤ c) : max3 a ≤ c := by
  exact max_le h0 (max_le h1 h2)

/-- A slot reduction is *safe* if the scalar slot constraint implies every resource
constraint.  The sufficient condition is `phi a ≥ max_r a_r`. -/
lemma safe_of_phi_ge (a p : Fin 3 → ℝ) (phi : (Fin 3 → ℝ) → ℝ)
    (ha : ∀ r, 0 ≤ a r) (hp : ∀ r, 0 ≤ p r)
    (hphi : ∀ r, a r ≤ phi a) (hbudget : ∑ r, phi a * p r ≤ 1) :
    ∑ r, a r * p r ≤ 1 := by
  have h1 : ∑ r, a r * p r ≤ ∑ r, phi a * p r := by
    apply Finset.sum_le_sum
    intro r hr
    nlinarith [hphi r, hp r]
  linarith

/-- If the slot weight is at most one, the cardinality budget implies the slot budget. -/
lemma safe_of_phi_le_one (a p : Fin 3 → ℝ) (phi : (Fin 3 → ℝ) → ℝ)
    (ha : ∀ r, 0 ≤ a r) (hp : ∀ r, 0 ≤ p r)
    (hphi : ∀ r, a r ≤ phi a) (hphi1 : phi a ≤ 1) (hbudget : ∑ r, p r ≤ 1) :
    ∑ r, a r * p r ≤ 1 := by
  apply safe_of_phi_ge a p phi ha hp hphi
  have hmul : phi a * ∑ r, p r ≤ phi a * 1 :=
    mul_le_mul_of_nonneg_left hbudget (le_trans (ha 0) (hphi 0))
  have hmul' : phi a * ∑ r, p r ≤ phi a := by simpa using hmul
  have hsum : ∑ r, phi a * p r = phi a * ∑ r, p r := by rw [Finset.mul_sum]
  rw [hsum]
  linarith

/-- Dominant-resource (max-norm) slots, equation (14). -/
def dominant (a : Fin 3 → ℝ) : ℝ := max3 a

lemma dominant_ge (a : Fin 3 → ℝ) (r : Fin 3) : a r ≤ dominant a := max3_ge a r

lemma dominant_safe (a p : Fin 3 → ℝ)
    (ha : ∀ r, 0 ≤ a r) (hp : ∀ r, 0 ≤ p r) (hbudget : ∑ r, dominant a * p r ≤ 1) :
    ∑ r, a r * p r ≤ 1 :=
  safe_of_phi_ge a p dominant ha hp (fun r => dominant_ge a r) hbudget

/-- Weighted linear scalarization, equation (15). -/
def linear (α a : Fin 3 → ℝ) : ℝ := ∑ r, α r * a r

lemma linear_ge_max (a α : Fin 3 → ℝ)
    (ha : ∀ r, 0 ≤ a r) (hα : ∀ r, 1 ≤ α r) : max3 a ≤ linear α a := by
  have hterm : ∀ r, a r ≤ α r * a r := by
    intro r
    nlinarith [ha r, hα r]
  have h0 : a 0 ≤ linear α a := by
    simp [linear, Fin.sum_univ_three]
    nlinarith [hterm 0, ha 1, ha 2, hα 1, hα 2]
  have h1 : a 1 ≤ linear α a := by
    simp [linear, Fin.sum_univ_three]
    nlinarith [hterm 1, ha 0, ha 2, hα 0, hα 2]
  have h2 : a 2 ≤ linear α a := by
    simp [linear, Fin.sum_univ_three]
    nlinarith [hterm 2, ha 0, ha 1, hα 0, hα 1]
  exact max3_le h0 h1 h2

lemma linear_safe (a p α : Fin 3 → ℝ)
    (ha : ∀ r, 0 ≤ a r) (hp : ∀ r, 0 ≤ p r) (hα : ∀ r, 1 ≤ α r)
    (hbudget : ∑ r, linear α a * p r ≤ 1) : ∑ r, a r * p r ≤ 1 :=
  safe_of_phi_ge a p (linear α) ha hp
    (fun r => le_trans (max3_ge a r) (linear_ge_max a α ha hα)) hbudget

/-- The `p = 1` endpoint of the ℓ_p family, equation (16). -/
def lp1 (a : Fin 3 → ℝ) : ℝ := ∑ r, a r

lemma lp1_eq_linear (a : Fin 3 → ℝ) : lp1 a = linear (fun _ => 1) a := by
  simp [lp1, linear]

/-- Capacity loss factor of a slot reduction, equation (17), for one service. -/
noncomputable def lambda (phi : (Fin 3 → ℝ) → ℝ) (a : Fin 3 → ℝ) : ℝ := phi a / max3 a

lemma lambda_ge_one (phi : (Fin 3 → ℝ) → ℝ) (a : Fin 3 → ℝ)
    (hmax : 0 < max3 a) (hphi : max3 a ≤ phi a) : 1 ≤ lambda phi a := by
  rw [lambda]
  exact (one_le_div hmax).2 hphi

/-- If `phi` is safe then its capacity loss is at least one. -/
lemma lambda_safe_ge_one (phi : (Fin 3 → ℝ) → ℝ) (a : Fin 3 → ℝ)
    (hmax : 0 < max3 a) (hphi : ∀ r, a r ≤ phi a) : 1 ≤ lambda phi a :=
  lambda_ge_one phi a hmax (max3_le (hphi 0) (hphi 1) (hphi 2))

end RandomizedDRL
