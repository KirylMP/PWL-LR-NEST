# Paper: scope and reproducibility

The current paper baseline consists of the six candidate model artifacts in
[`models/paper_frozen_v1`](../models/paper_frozen_v1/manifest.json). The
manifest records their source and SHA-256 hashes. This is a candidate snapshot,
not a declaration that every experiment or claim is ready for publication.
The original code commit in the manifest refers to the upstream repository;
the baseline commit in this repository is `31326122cfaff28c7c743851fbb21adfc1da18e6`.

The legacy copy in `astrocyte_lab/paper_models/` is still required by historical
scripts. Do not edit either copy in place. Check both with:

```bash
python scripts/verify_paper_snapshot.py
```

This checks file hashes, not scientific validity, execution environment, or
reproducibility of the computations. The README documents the limited 450 s
check already performed and the remaining work.

## Selecting material for the paper

The exact figures, claims and experiments are not yet fixed. For each selected
result, record here or in a versioned file: the claim and figure/table, input
model and hashes, code commit, command and configuration, environment and
dependencies, random seeds, output location, and validation and limitations.
Check that outputs can be regenerated before citing them. Pin a repository
commit or release when the actual manuscript snapshot is agreed. A later
candidate should get a new versioned snapshot and manifest rather than silently
changing `paper_frozen_v1`.
