# Server run: AM / FM / AFM

The experiment is self-contained. Original baseline and validation inputs are copies from PWL-LR-NEST main commit 416f56d8959990a1eeaef1a35ae6526f9b1ebeef. No original repository files are modified. Python 3.10+, NumPy, SciPy, Matplotlib required.

```
python -m venv .venv
. .venv/bin/activate
pip install numpy scipy matplotlib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python fit_server.py --case AM --seed 0 --max-nfev 300 --out runs/AM/0
python summarize_server.py runs
```

Independent multi-start jobs (example 4 simultaneous processes):

```
mkdir -p runs
for case in AM FM AFM; do for seed in $(seq 0 11); do printf '%s %s\n' "$case" "$seed"; done; done > jobs.txt
cat jobs.txt | xargs -n 2 -P 4 sh -c 'python fit_server.py --case "$0" --seed "$1" --max-nfev 300 --out "runs/$0/$1" > "runs/$0-$1.log" 2>&1'
python summarize_server.py runs
```

Each job writes its own checkpoint.json (best evaluated point, possibly from a finite-difference probe) and result.json. Never use the same --out directory concurrently. A normal result also starts from the best evaluated point, not necessarily optimizer's last iterate. --max-seconds is cooperative, checked after residual evaluation; reference preparation and validation are outside it. Job interruptions may leave only the checkpoint. Use command line --help for start type, jitter, sampling, and time limit. Record versions and commands. Number of finite-difference evaluations can substantially exceed --max-nfev; a single trajectory objective may take several seconds.

Optimization: exact-continuity basis F=[1,C,h,I,(C-C0)+,(C-g(I))+] @ coefficients, with g(I)=C0+height*clip((I-I1)/(I2-I1),0,1). Fits sampled C,h over one period from the LR orbit's peak state, includes closure penalty and LR-equilibrium residuals. Trace/determinant/discriminant penalties favor stable lower and upper fields and an unstable-focus middle field; upper focus is an additional modeling hypothesis. They are soft, and `spectrum_margin_pass` tests whether the numerical margins were achieved. Seeds alternate between the frozen physical model approximation and exploratory field fit. Geometry and coefficients are optimized simultaneously with least_squares; the reference peak state is not optimized. Local minima, nonsmooth threshold changes and failed orbits can impede convergence. Consider more starts and adjust weights only after documenting the rationale.

Validation integrates each candidate from the original reference initial conditions for max(450 s,14 LR periods) and heuristically detects converged attracting cycles. Four trajectories per case are held out from trajectory fitting, but equilibrium penalties cover their parameter range: this is NOT holdout of I from all training. Candidate period/amplitude errors are reported only where cycles were detected; do not rank candidates by error alone when cycles are missing. See result.json for all status values and region eigenvalues.

A spectral margin pass is NOT verification of an HLB1: locate the equilibrium and verify that it actually crosses the relevant boundary, determine the spectral types on both sides and check local nondegeneracy / return map. Likewise forward attracting cycles do not establish unstable branches or a fold. Check closure, multipliers, nondegeneracy and continuation using an independent solver. Verify boundaries and integrations at tighter tolerances, out-of-sample I values and longer transients. Model performance is not benchmarked and NEST is not tested.

Smoke test: AFM seed 0, `--max-nfev 1 --samples 12`; exercises reference generation, optimization call, and full validation, not convergence. Stored in smoke/afm/. The preliminary unconstrained starting fits are stored in seeds/.

## Returning results through Git

The repository ignores `runs/`, so add only the outputs you want to retain explicitly. From the experiment directory, after the runs finish:

```bash
python summarize_server.py runs | tee runs/summary.txt
python -m pip freeze > runs/environment.txt
git rev-parse HEAD > runs/code-commit.txt
git add -f runs/summary.txt runs/environment.txt runs/code-commit.txt
find runs -type f \( -name result.json -o -name checkpoint.json -o -name "*.log" \) -print0 | xargs -0 -r git add -f --
git status
git commit -m "Record sloped upper threshold experiment runs"
git push
```

Run on a dedicated branch if desired. Check `git status` and output sizes before committing. Store the command, job count and any non-default settings with the results. A checkpoint alone is not a completed run.
