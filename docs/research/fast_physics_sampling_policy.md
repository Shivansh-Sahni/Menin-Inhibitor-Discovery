# Fast-physics sampling policy

This layer is a deterministic hypothesis screen, not an equilibrium-population calculation. Approximate pKa evidence, one ETKDG seed, MMFF94s/UFF minimization, finite clustering, and the absence of explicit solvent remain material limitations. **All local all-series structure, microstate, and conformer execution is currently deferred.** The rung definitions below are future-work contracts and do not authorize modeling from an unexecuted 25-conformer screen.

## Chemical-state inclusion

- The nominal inclusion threshold is **0.1%** (`0.001`) at the maximum nominal estimated population across modeled pH values.
- States between **0.01% and 0.1%** are retained as sensitivity candidates when the compute cap permits. This protects mechanisms in which a rare state has disproportionate transport or binding propensity.
- Every run reports counterfactual threshold effects at **1%, 0.1%, and 0.01%**, including state counts, probability mass below each threshold, actual retained mass, and any threshold-qualifying mass lost to the compute cap.
- Reference tautomers and representatives of net charge, neutral character, zwitterionic character, and unique charge/H-bond-capacity signatures are retained as structural exceptions. HBD/HBA capacity is only a pre-conformer proxy for exposure; the subsequent 3D ensemble supplies the exposure descriptors.
- `max_states=24` is only a prospective local compute cap, not a desired or physically privileged state count. If it removes any population-qualifying or structural-exception state, the structure is marked substantively inadmissible rather than silently treated as complete.
- HPC enumeration does **not** inherit the 24-state local cap. It must enumerate every state that passes the 0.01% sensitivity floor or a declared structural-exception rule, then document any separate resource limit and omitted mass explicitly.

The population values are Henderson–Hasselbalch sensitivity weights, not measured microscopic populations. Omitted mass and threshold stability must therefore be interpreted together with the ±1 pKa scenarios.

## Conformer-depth audit

If resumed, the time-bounded all-series generated depth is **25 conformers per retained state**, and up to 25 minimized cluster representatives are retained. This is the first future rung of the declared convergence ladder, not an assertion that 25 is sufficient. A nested audit uses deterministic rank prefixes of the retained ensemble and compares weighted 3D polar SASA, radius of gyration, IMHB count, minimum energy, and effective conformer count. The latest two subsets pass the internal-stability screen only when:

- polar-SASA and radius-of-gyration relative changes are each at most 10%;
- mean IMHB-count change is at most 0.5;
- minimum-energy change is at most 1.0 kcal/mol; and
- effective conformer count is at least 1.5 unless only one conformer exists.

These nested prefixes are correlated subsets of one deterministic pool. Passing is not proof of convergence, and failure is an escalation signal rather than evidence that the mechanism is false.

The audit abstains whenever retained depth is smaller than generated depth. A 50-, 100-, 250-, or 500-conformer generation run cannot inherit a stability conclusion from only 25 retained conformers; the same generated pool must be retained through the audited depth or evaluated by a separately justified coverage diagnostic.

- **50, 100, and 250** are selected-state escalation rungs for unstable or claim-critical states.
- **500** is the selected-pilot comparator when the 250-depth result remains unstable.
- Neither 25, 50, 100, 250, nor 500 is a universal or validated sampling constant. Persistent instability at 500 triggers method review, additional seeds, or enhanced/explicit-solvent sampling rather than automatic brute-force expansion.

The canonical outputs are `fast_physics_state_threshold_audit.parquet` and `fast_physics_sampling_escalation_queue.parquet`.
