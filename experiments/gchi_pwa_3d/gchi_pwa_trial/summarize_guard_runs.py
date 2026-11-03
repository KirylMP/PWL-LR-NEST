#!/usr/bin/env python3
"""Compact table; conditional metrics are never a standalone selection rule."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
print('seed objective Cb k max_anchor_res guard_max_real cycles/25 below_H1 end_dist sweep_cycles/total sweep_unknown')
for directory in sorted([d for d in a.directory.iterdir() if d.is_dir()],key=lambda p:p.name):
 cand=directory/'candidate.json';val=directory/'validation.json'
 if not cand.exists():print(directory.name,'NO_CANDIDATE (inspect log/checkpoint)');continue
 d=json.loads(cand.read_text());v=json.loads(val.read_text()) if val.exists() else None
 maxanchor=max(z['residual'] for z in d['fixed_equilibria']);guard=max(z[0] for z in v['guard_eigenvalues']) if v else None
 eq=v['below_H1']['equilibria'] if v else [];end=v['below_H1']['trajectory'].get('end') if v else None
 distance=float(np.linalg.norm(np.asarray(end)-eq[0]['state'])) if len(eq)==1 and end is not None else None
 probes=v.get('near_h1_sweep',{}).get('probes',[]) if v else [];sc=sum(p['status']=='cycle' for p in probes);unknown=sum(p['status'] not in ('cycle','near_equilibrium') for p in probes)
 print(d['seed'],f"{d['best_loss']:.6g}",f"{d['barrier']['Cb']:.6f}",f"{d['barrier']['k']:.5f}",f'{maxanchor:.2e}',f'{guard:.5g}' if guard is not None else '-',f"{v['detected_cycles']}/{v['total']}" if v else '-',v['below_H1']['trajectory']['status'] if v else '-',f'{distance:.3g}' if distance is not None else '-', f'{sc}/{len(probes)}' if probes else '-', unknown if probes else '-')
