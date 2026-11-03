#!/usr/bin/env python3
"""Build NEST 3.10 models from the two selected 3D G-ChI candidates."""
import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UPSTREAM = ROOT / 'experiments/fm_am_minimal_zones/nest_benchmark/module/upstream'
MODES = HERE.parent / 'gchi_modes'
DEFAULTS = {'am': MODES / 'runs/afm-calcium-c1-01/candidate-2.json',
            'fm': MODES / 'runs/fm-topology-free-c1-01/candidate-3.json'}


def run_config(option):
    return subprocess.check_output(['nest-config', option], text=True).strip()


def build_source(case, path, out):
    raw = path.read_bytes()
    d = json.loads(raw)
    co = np.asarray(d['coefficients'], dtype=float)
    knots = np.asarray(d['knots'], dtype=float)
    if d['mode'] != {'am': 'AFM', 'fm': 'FM'}[case] or co.shape != (10, 3) or knots.shape != (3,):
        raise ValueError(f'{path}: unexpected model, basis or coefficient dimensions')
    if not np.all(np.isfinite(co)) or not np.all(np.isfinite(knots)) or not np.all(np.diff(knots) > 0):
        raise ValueError(f'{path}: invalid coefficients or knots')
    name = 'gchi_' + case + '_pwa'
    literal = lambda a: ','.join(format(float(x), '.17g') for x in np.ravel(a))
    for ext in ('h', 'cpp'):
        src = (UPSTREAM / f'astrocyte_lr_1994.{ext}').read_text()
        src = src.replace('astrocyte_lr_1994', name).replace('ASTROCYTE_LR_1994', name.upper())
        if ext == 'cpp':
            start = src.index('  // a shorthand', src.index(name + '_dynamics('))
            end = src.index('  return GSL_SUCCESS;', start)
            body = '''  typedef nest::NAME::State_ S;
  const nest::NAME& node = *(reinterpret_cast<nest::NAME*>(pnode));
  const double C = y[S::Ca_astro], h = y[S::h_IP3R], I = y[S::IP3];
  // IP3_0 is repurposed as fixed extracellular glutamate gamma (uM).
  const double g = node.P_.IP3_0_;
  const double p = std::pow(g, 0.7);
  const double z[10] = {1., C, h, I, p/(p+std::pow(.02,.7)),
    p/(p+std::pow(.5,.7)), p/(p+std::pow(3.,.7)),
    std::max(C-KNOTS[0],0.), std::max(C-KNOTS[1],0.), std::max(C-KNOTS[2],0.)};
  double result[3] = {0., 0., 0.};
  for (int j=0; j<10; ++j) for (int k=0; k<3; ++k) result[k] += z[j]*CO[j][k];
  // Fitted rates are per second; NEST integrates in milliseconds.
  f[S::Ca_astro] = result[0] / 1000.;
  f[S::h_IP3R] = result[1] / 1000.;
  f[S::IP3] = result[2] / 1000.;

'''.replace('NAME', name)
            src = src[:start] + body + src[end:]
            src = src.replace('#include "' + name + '.h"', '#include "' + name + '.h"\n'
                              + 'static const double KNOTS[3] = {' + literal(knots) + '};\n'
                              + 'static const double CO[10][3] = {'
                              + ','.join('{' + literal(row) + '}' for row in co) + '};')
            clamp = '    S_.y_[ State_::Ca_astro ] = std::max( 0.0, std::min( S_.y_[ State_::Ca_astro ], P_.Ca_tot_ ) );'
            spike = '    S_.y_[ State_::IP3 ] += P_.delta_IP3_ * B_.spike_exc_.get_value( lag );'
            if clamp not in src or spike not in src:
                raise RuntimeError('Upstream template changed')
            src = src.replace(clamp, '    // No state clipping: physical violations remain observable.')
            src = src.replace(spike, '    // No spike-driven IP3 in this fixed-glutamate experiment.')
            src = src.replace('gsl_odeiv_control_y_new( 1e-3, 0.0 )', 'gsl_odeiv_control_y_new( 1e-10, 1e-9 )')
            src = src.replace('gsl_odeiv_control_init( B_.c_, 1e-3, 0.0, 1.0, 0.0 )', 'gsl_odeiv_control_init( B_.c_, 1e-10, 1e-9, 1.0, 0.0 )')
        (out / f'{name}.{ext}').write_text(src)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--am', type=Path, default=DEFAULTS['am'])
    ap.add_argument('--fm', type=Path, default=DEFAULTS['fm'])
    args = ap.parse_args()
    out = HERE / 'build'
    out.mkdir(exist_ok=True)
    manifest = {case: build_source(case, getattr(args, case), out) for case in ('am', 'fm')}
    (out / 'module.cpp').write_text('''#include "nest_extension_interface.h"
#include <string>
namespace nest { void register_gchi_am_pwa(const std::string&); void register_gchi_fm_pwa(const std::string&); }
class GchiModule : public nest::NESTExtensionInterface { public: void initialize() override {
  nest::register_gchi_am_pwa("gchi_am_pwa"); nest::register_gchi_fm_pwa("gchi_fm_pwa"); } };
GchiModule gchi_pwa_module_LTX_module;
''')
    compiler = shlex.split(run_config('--compiler'))
    cmd = (compiler + ['-std=c++17', '-O2', '-shared', '-fPIC']
           + shlex.split(run_config('--cflags')) + shlex.split(run_config('--includes'))
           + [str(out / p) for p in ('module.cpp', 'gchi_am_pwa.cpp', 'gchi_fm_pwa.cpp')]
           + ['-o', str(out / 'gchi_pwa_module.so')] + shlex.split(run_config('--libs')) + ['-lgsl', '-lgslcblas'])
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Built', out / 'gchi_pwa_module.so')


if __name__ == '__main__':
    main()
