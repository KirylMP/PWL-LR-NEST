#!/usr/bin/env python3
"""Build NEST 3.10 models from the two selected 3D G-ChI candidates."""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
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


def detect_cxx11_abi():
    import nest.nestkernel_api as kernel
    output = subprocess.check_output(['nm', '-D', '-C', kernel.__file__], text=True)
    symbols = [s for s in output.splitlines() if ' T nest::StructuralPlasticityNode::get_synaptic_elements(' in s]
    if not symbols:
        raise RuntimeError('Could not detect NEST C++ ABI; pass --cxx11-abi 0 or 1')
    return int(any('std::__cxx11::' in s or '[abi:cxx11]' in s for s in symbols))


def compiler_command(requested):
    options = [requested, os.environ.get('CXX'), run_config('--compiler'), 'g++', 'c++']
    for option in options:
        if not option:
            continue
        parts = shlex.split(option)
        if parts and (shutil.which(parts[0]) or Path(parts[0]).is_file()):
            return parts
    raise RuntimeError('C++ compiler not found; install g++ or pass --cxx PATH')


def nest_flags(extra_include=None):
    # Binary wheels may embed build-host paths in nest-config. Locate the
    # installed headers relative to PyNEST and omit vanished library paths.
    import nest
    platlib = Path(nest.__file__).resolve().parent.parent
    includes = []
    if extra_include:
        if not (Path(extra_include) / 'nest_extension_interface.h').is_file():
            raise RuntimeError('--nest-include must point to the directory containing nest_extension_interface.h')
        includes.append('-I' + str(Path(extra_include).resolve()))
    for flag in shlex.split(run_config('--includes')):
        if flag.startswith('-I') and Path(flag[2:]).is_dir():
            includes.append(flag)
    for p in (Path(nest.__file__).resolve().parent / 'include/nest',
              platlib / 'include/nest', Path(sys.prefix) / 'include/nest'):
        if p.is_dir():
            includes.append('-I' + str(p))
    if not any((Path(x[2:]) / 'nest_extension_interface.h').is_file() for x in includes):
        raise RuntimeError('NEST development headers not found. Install NEST headers or fix nest-config --includes.')
    libs = []
    for flag in shlex.split(run_config('--libs')):
        if flag.startswith('-L') and not Path(flag[2:]).is_dir():
            continue
        if flag.startswith('/') and not Path(flag).exists():
            continue
        libs.append(flag)
    for p in (Path(nest.__file__).resolve().parent / 'lib64/nest',
              Path(nest.__file__).resolve().parent / 'lib/nest',
              platlib / 'lib64/nest', platlib / 'lib/nest'):
        if p.is_dir():
            libs.append('-L' + str(p))
    return includes, libs


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
    ap.add_argument('--cxx', help='C++ compiler (default: CXX, nest-config, then g++)')
    ap.add_argument('--nest-include', type=Path, help='directory containing nest_extension_interface.h')
    ap.add_argument('--cxx11-abi', type=int, choices=(0, 1), help='override automatic NEST libstdc++ ABI detection')
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
    compiler = compiler_command(args.cxx)
    includes, libs = nest_flags(args.nest_include)
    abi = detect_cxx11_abi() if args.cxx11_abi is None else args.cxx11_abi
    print('NEST _GLIBCXX_USE_CXX11_ABI =', abi, flush=True)
    cmd = (compiler + ['-std=c++17', '-O2', '-shared', '-fPIC']
           + shlex.split(run_config('--cflags')) + ['-U_GLIBCXX_USE_CXX11_ABI', f'-D_GLIBCXX_USE_CXX11_ABI={abi}'] + includes
           + [str(out / p) for p in ('module.cpp', 'gchi_am_pwa.cpp', 'gchi_fm_pwa.cpp')]
           + ['-o', str(out / 'gchi_pwa_module.so')] + libs + ['-lgsl', '-lgslcblas'])
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Built', out / 'gchi_pwa_module.so')


if __name__ == '__main__':
    main()
