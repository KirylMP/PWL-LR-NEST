#!/usr/bin/env python3
"""Compare NEST 3D PWA trajectories to SciPy PWA and original G-ChI."""
import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
MODES = HERE.parent / 'gchi_modes'
sys.path.insert(0, str(MODES))
from fit_modes import design
from explore_modes import P, field


def rhs(co, knots, gamma):
    return lambda t, y: design(y, gamma, knots)[0] @ co


def metrics(y):
    tail = y[:, len(y[0]) // 2:]
    return {'min_C_all': float(y[0].min()), 'min_I_all': float(y[2].min()),
            'min_h_all': float(y[1].min()), 'max_h_all': float(y[1].max()),
            'C_tail_min': float(tail[0].min()), 'C_tail_max': float(tail[0].max()),
            'I_tail_min': float(tail[2].min()), 'I_tail_max': float(tail[2].max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=200., help='seconds')
    ap.add_argument('--resolution', type=float, default=.1, help='NEST resolution, ms')
    ap.add_argument('--interval', type=float, default=2., help='recording interval, ms')
    ap.add_argument('--samples', type=int, default=3, help='number of G-ChI cycle inputs per mode')
    ap.add_argument('--out', type=Path, default=HERE / 'results/analytic')
    args = ap.parse_args()
    if args.duration <= 0 or args.resolution <= 0 or args.samples < 1 or args.interval < args.resolution:
        ap.error('duration, resolution and samples must be positive; interval must be >= resolution')
    if not np.isclose(args.interval / args.resolution, round(args.interval / args.resolution)):
        ap.error('interval must be a multiple of resolution')
    import nest
    import nest.nestkernel_api as nest_kernel
    # Pip wheels may contain the kernel inside this extension rather than a
    # separate libnest.so; make its symbols visible to the NEST module loader.
    kernel_handle = ctypes.CDLL(nest_kernel.__file__, mode=ctypes.RTLD_GLOBAL)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    module = HERE / 'build/gchi_pwa_module.so'
    if not module.is_file():
        raise FileNotFoundError(f'{module} not found; run python build.py first')
    manifest = json.loads((HERE / 'build/manifest.json').read_text())
    for x in manifest.values():
        if hashlib.sha256(Path(x['path']).read_bytes()).hexdigest() != x['sha256']:
            raise RuntimeError('Candidate changed after module build; rebuild first')
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for case, mode in (('am', 'AFM'), ('fm', 'FM')):
        data = json.loads(Path(manifest[case]['path']).read_text())
        co, knots = np.array(data['coefficients']), np.array(data['knots'])
        reference = json.loads((MODES / 'results' / mode / 'reference.json').read_text())
        cycles = [v for v in reference['cycles'] if v['status'] == 'cycle' and 'peak' in v]
        indices = np.unique(np.linspace(0, len(cycles)-1, min(args.samples, len(cycles)), dtype=int))
        if len(indices) == 0:
            raise RuntimeError('No reference cycles available')
        for idx in indices:
            sample = cycles[idx]
            g, initial = float(sample['gamma']), np.asarray(sample['peak'], dtype=float)
            nest.ResetKernel()
            nest.SetKernelStatus({'resolution': args.resolution, 'local_num_threads': 1, 'print_time': False})
            nest.Install(str(module))
            node = nest.Create('gchi_' + case + '_analytic', params={'IP3_0': g, 'Ca_astro': float(initial[0]),
                                'h_IP3R': float(initial[1]), 'IP3': float(initial[2])})
            meter = nest.Create('multimeter', params={'interval': args.interval,
                                'record_from': ['Ca_astro', 'h_IP3R', 'IP3']})
            nest.Connect(meter, node)
            nest.Simulate(args.duration * 1000.)
            ev = meter.get('events')
            t = np.asarray(ev['times'], dtype=float) / 1000.
            y = np.array([ev['Ca_astro'], ev['h_IP3R'], ev['IP3']], dtype=float)
            if len(t) < 2 or np.any(np.diff(t) <= 0):
                raise RuntimeError('Invalid NEST record times')
            kw = dict(t_eval=t, method='DOP853', rtol=1e-10, atol=1e-12, max_step=.05)
            a = solve_ivp(rhs(co, knots, g), (0., args.duration), initial, **kw)
            b = solve_ivp(field, (0., args.duration), initial, args=(g, P[mode]), **kw)
            if not a.success or not b.success:
                raise RuntimeError('Independent integration failed')
            row = {'mode': mode, 'gamma': g, 'initial': initial.tolist(), 'nest': metrics(y),
                   'scipy_pwa': metrics(a.y), 'gchi': metrics(b.y),
                   'nest_scipy_max_abs': np.max(np.abs(y - a.y), axis=1).tolist()}
            records.append(row)
            stem = f'{mode.lower()}-{g:.6g}'
            np.savez_compressed(args.out / (stem + '.npz'), time_s=t, nest=y, scipy_pwa=a.y, gchi=b.y)
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            for j, (ax, label) in enumerate(zip(axes, ('C (uM)', 'h', 'IP3 (uM)'))):
                ax.plot(t, y[j], label='NEST analytic PWA', lw=1)
                ax.plot(t, a.y[j], '--', label='SciPy PWA', lw=.8)
                ax.plot(t, b.y[j], ':', label='G-ChI', lw=.8)
                ax.set_ylabel(label); ax.grid(alpha=.25)
            axes[0].set_title(f'{mode}, glutamate = {g:.6g} uM')
            axes[0].legend(); axes[-1].set_xlabel('time (s)')
            fig.tight_layout(); fig.savefig(args.out / (stem + '.png'), dpi=140); plt.close(fig)
            print(mode, 'gamma', g, 'NEST vs SciPy max', row['nest_scipy_max_abs'], flush=True)
    result = {'nest_version': str(getattr(nest, '__version__', 'unknown')), 'duration_s': args.duration,
              'resolution_ms': args.resolution, 'record_interval_ms': args.interval, 'manifest': manifest,
              'records': records, 'limitations': 'Sampled fixed-glutamate trajectories only; no global positivity, fold, or speed claim. Phase errors may accumulate in long oscillatory traces.'}
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('WROTE', args.out)


if __name__ == '__main__':
    main()
