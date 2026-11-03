# Recheck of FM server run

Input candidates are `../../runs/fm-search-01/candidate-{2,8,10}.json` from commit 8e65ceef. Recheck uses the updated validator: IP3 >= 0, Ca >= 0, and 0 <= h <= 1 at solver nodes across the full trajectory and at sampled points on a detected cycle. Starting from a reference cycle point, seed 2 has 7/12 valid integrated trajectories, seed 8 has 3/12, seed 10 has 5/12; AFM local candidate still has 12/12. A trajectory may be rejected for a transient violation even if its eventual cycle is physical. Neither sample checking nor return from a single initial state proves physical invariance, orbital stability, or absence of other attractors.

Previously reported 9/12 and 7/12 for seeds 2 and 10 used a validator that did not reject negative IP3. Seed 10 contains negative IP3 on the detected periodic orbit at gamma=.22183 and .43578. Among the corrected seven for seed 2, mean absolute Cmax and period discrepancies are .42948 uM and 5.6516 s (conditional on valid detected cycles).

The threshold-intersection search finds that seed 2's stable zero-glutamate equilibrium branch crosses the lowest threshold at gamma=.0103157, with leading eigenvalue jumping from -.12683 to +4.62678. Thus a loss of stability occurs there, earlier than the pinned reference lower transition at gamma=.0201525. Exact pinning only guarantees that the latter point is an equilibrium on a switching boundary; it does not ensure that the branch issuing from the rest state reaches that point. Further intersections exist. Search uses a finite grid and may miss roots; it does not establish HLB subtype.

To reproduce a recheck from `gchi_modes`:

```bash
python validate_modes.py --reference results/FM/reference.json --candidate runs/fm-search-01/candidate-2.json --output recheck-2.json
python analyze_equilibria.py --candidate runs/fm-search-01/candidate-2.json --output intersections-2.json
```
