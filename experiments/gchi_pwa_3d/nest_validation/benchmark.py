#!/usr/bin/env python3
"""Timed NEST runs of PWA and reference G-ChI using the same GSL solver."""
import argparse
import ctypes
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
MODES = HERE.parent / 'gchi_modes'
sys.path.insert(0, str(MODES))
from explore_modes import field, P
from fit_modes import design


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', type=int, nargs='+', default=[1, 100, 1000])
    ap.add_argument('--duration', type=float, default=5., help='timed simulated seconds')
    ap.add_argument('--warmup', type=float, default=1., help='simulated seconds before timing')
    ap.add_argument('--resolution', type=float, default=.1, help='NEST step in ms')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', type=Path, default=HERE / 'results/speed.json')
    args = ap.parse_args()
    if min(args.nodes) < 1 or min(args.duration, args.resolution) <= 0 or args.warmup < 0 or min(args.repeats, args.threads) < 1:
        ap.error('nodes, duration, resolution, repeats and threads must be positive; warmup must be nonnegative')
    for seconds in (args.warmup, args.duration):
        steps = seconds * 1000 / args.resolution
        if not np.isclose(steps, round(steps)):
            ap.error('duration and warmup must be multiples of NEST resolution')
    import nest
    import nest.nestkernel_api as kernel
    kernel_handle = ctypes.CDLL(kernel.__file__, mode=ctypes.RTLD_GLOBAL)
    module = HERE / 'build/gchi_pwa_module.so'
    manifest = json.loads((HERE / 'build/manifest.json').read_text())
    import hashlib
    for value in manifest.values():
        if hashlib.sha256(Path(value['path']).read_bytes()).hexdigest() != value['sha256']:
            raise RuntimeError('Candidate changed after building; rebuild')
    scenarios = {}
    for case, mode in (('am', 'AFM'), ('fm', 'FM')):
        reference = json.loads((MODES / 'results' / mode / 'reference.json').read_text())
        cycles = [v for v in reference['cycles'] if v['status'] == 'cycle' and 'peak' in v]
        point = cycles[len(cycles)//2]
        scenarios[case] = {'mode': mode, 'gamma': point['gamma'], 'initial': point['peak']}

    def setup(case, which, count):
        s = scenarios[case]
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': args.resolution, 'local_num_threads': args.threads, 'print_time': False})
        nest.Install(str(module))
        c, h, i = s['initial']
        return nest.Create(f'gchi_{case}_{which}', n=count, params={'IP3_0': s['gamma'],
                           'Ca_astro': c, 'h_IP3R': h, 'IP3': i})

    # Verify both generated models against independent SciPy at the final state.
    checks = []
    for case, s in scenarios.items():
        data = json.loads(Path(manifest[case]['path']).read_text())
        co, knots = np.array(data['coefficients']), np.array(data['knots'])
        for which in ('pwa', 'reference'):
            node = setup(case, which, 1)
            nest.Simulate(2000.)
            actual = np.array([nest.GetStatus(node, name)[0] for name in ('Ca_astro', 'h_IP3R', 'IP3')])
            rhs = (lambda t, y: design(y, s['gamma'], knots)[0] @ co) if which == 'pwa' else (lambda t, y: field(t, y, s['gamma'], P[s['mode']]))
            sol = solve_ivp(rhs, (0., 2.), s['initial'], method='DOP853', rtol=1e-11, atol=1e-13, max_step=.01)
            error = float(np.max(np.abs(actual - sol.y[:, -1])))
            checks.append({'mode': case, 'model': which, 'max_abs': error})
            if not sol.success or error > 1e-4:
                raise RuntimeError(f'Correctness precheck failed: {case} {which} max error {error}')

    tasks = [(case, which, count, repeat) for case in scenarios for which in ('pwa', 'reference')
             for count in args.nodes for repeat in range(args.repeats)]
    random.Random(20261103).shuffle(tasks)
    results = []
    for case, which, count, repeat in tasks:
        setup(case, which, count)
        nest.Prepare()
        try:
            if args.warmup:
                nest.Run(args.warmup * 1000.)
            start = time.perf_counter()
            nest.Run(args.duration * 1000.)
            wall = time.perf_counter() - start
        finally:
            nest.Cleanup()
        row = {'mode': case, 'model': which, 'nodes': count, 'repeat': repeat, 'wall_seconds': wall}
        results.append(row)
        print(case, which, count, repeat, 'wall_s', round(wall, 6), flush=True)
    summary = []
    for case in scenarios:
        for n in args.nodes:
            times = {which: [x['wall_seconds'] for x in results if x['mode'] == case and x['model'] == which and x['nodes'] == n] for which in ('pwa', 'reference')}
            pwa, ref = statistics.median(times['pwa']), statistics.median(times['reference'])
            summary.append({'mode': case, 'nodes': n, 'pwa_median_s': pwa, 'reference_median_s': ref, 'speedup_reference_over_pwa': ref / pwa})
    result = {'nest_version': str(getattr(nest, '__version__', 'unknown')), 'platform': platform.platform(),
              'processor': platform.processor(), 'python': sys.version, 'settings': vars(args) | {'out': str(args.out)},
              'manifest': manifest, 'scenarios': scenarios, 'prechecks': checks, 'runs': results, 'summary': summary,
              'scope': 'Fixed glutamate, independent identical nodes, no recording/connections. Both use adaptive GSL RKF45 with identical tolerances. PWA is not an exact event solver.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for row in summary:
        print('SUMMARY', row, flush=True)
    print('WROTE', args.out)


if __name__ == '__main__':
    main()
