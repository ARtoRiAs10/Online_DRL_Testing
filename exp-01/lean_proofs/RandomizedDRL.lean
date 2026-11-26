import SlotReduction
import Projection
import Landlord
import Latency
import R3
import Regret

/-!
# Machine-checked algebra for the randomized multi-resource DRL draft

This root file collects the finite-dimensional lemmas used in the report:

* `SlotReduction`: equations (13)-(17), safe slot reductions and capacity loss.
* `Projection`: equations (20)-(24), the KKT threshold, the fixed-primal dual
  identity and the final rescaling.
* `Landlord`: equation (25), the residual-deficit score and its after-eviction form.
* `Latency`: equations (27)-(34), physical/expected latency and resource metrics.
* `R3`: the randomized ski-rental CDF and its inverse.
* `Regret`: the diameter and gradient ingredients of Proposition 1.

The full OGD regret inequality and the general `ℓ_p` bound for arbitrary `p` are
discussed in the report; they are not claimed as machine-checked here.
-/
namespace RandomizedDRL

/-- The slot constraint is feasible for the dominant-resource reduction. -/
theorem dominant_slot_feasible (a p : Fin 3 → ℝ)
    (ha : ∀ r, 0 ≤ a r) (hp : ∀ r, 0 ≤ p r)
    (hbudget : ∑ r, dominant a * p r ≤ 1) : ∑ r, a r * p r ≤ 1 :=
  dominant_safe a p ha hp hbudget

end RandomizedDRL
