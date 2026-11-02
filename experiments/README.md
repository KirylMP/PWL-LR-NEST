# Experiments

Use a separate directory under `experiments/` for each new investigation.
The existing `astrocyte_lab/experiments/` and `fm_am_minimal_zones/` contain
historical work and may also be dependencies of existing scripts; their location
does not determine whether a result belongs in the paper. They have not been
moved. Shared code currently lives in `src/pwl_nest/`, but some scripts also
depend on other experiment directories. Preserve and document these dependencies.

For a new experiment, copy `TEMPLATE.md` into its directory as `README.md`.
Commit the small scripts, configurations and provenance needed to rerun it.
Write generated outputs to `runs/` or `results/`, which are ignored by Git;
save important outputs separately and record their location and checksums. Do
not overwrite the paper snapshot or use it as an output directory. Results only
become paper material after explicit selection and validation; see
[`paper/README.md`](../paper/README.md).
