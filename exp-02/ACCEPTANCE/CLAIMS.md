# Claim-by-claim acceptance criteria

Frozen before running.  "Accept" requires the stated evidence; "reject or
downgrade" is the default when the evidence is missing or mixed.

| Claim | Accept if | Reject or downgrade if |
|---|---|---|
| **R3 improves the trigger (ratio 2 → e/(e−1))** | Pure ski-rental worst ratio is **1.58 ± 0.02** (already met). | The claim is only valid for the trigger in isolation. Do not accept it as an end-to-end ratio. |
| **R3 beats deterministic at equal cost** | At **3–4 matched cost levels** across the `lam` sweep, R3's latency is lower in at least **3 of 4**, with the **95% CI of the gap excluding 0**, on at least **2 of 3 datasets** (Google, SHARCNet, synthetic). | Latency gap CI includes 0, or R3 wins only at `lam` values where cost is unrealistically high. |
| **MG breaks eviction lockstep** | Latency at least **20% lower** on the cyclic adversary (the demo showed about 66%), and a paired test gives **p < 0.05** over at least **30 seeds**. | No gain over deterministic on adversarial patterns. |
| **MG does no harm on real traces** | Latency within about **1–2%** of deterministic Online-DRL on Google and SHARCNet, and cost not higher by more than the same margin. | A regression above **2%** on the real traces. |
| **R3+MG lowers E[ALG]/OPT_t** | On the Large Sequence, the ratio is lower for **every k in {6, 12, 24, 48}**, and the slope of ratio vs k is visibly flatter. | The ratio drops only at one k, or the slope is unchanged. The demo's 8.37 → 7.68 at k=6 is far from enough alone. |
| **Improves toward O(log k)** | The ratio grows roughly logarithmically in k across at least **4 values of k**, with a fitted slope clearly below the deterministic fit. | Treat as unsupported. This stays a conjecture without a proof, even with good curves. |
| **Overhead stays small** | Measured **µs/request** within about **2×** of deterministic, on identical hardware. | Larger overhead. The "9–28 µs" figure from earlier notes is not from the paper and must not be cited. |

## Decision rule

A claim is **accepted** only if every clause in its "Accept if" cell holds.
If any clause fails, the claim is **rejected** or **downgraded** to the weaker
statement in the "Reject or downgrade if" cell.  A claim that depends on a
failed precondition is **blocked**, regardless of its own evidence.