#!/usr/bin/env python3
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);print('seed cost cycles/12 Cmax_MAE Imax_MAE onset_0203_prev closure fold_det lower1_eig lower2_eig pair_detected optimizer_success perturbed')
def f(x):return None if x is None else round(x,6)
def sci(x):return None if x is None else f'{x:.2e}'
for path in sorted(p.glob('candidate-*.json'),key=lambda x:int(x.stem.split('-')[-1])):
 i=path.stem.split('-')[-1];names=['refinement','validation','onset','return-check'];paths=[p/f'{n}-{i}.json' for n in names]
 if not all(x.exists() for x in paths):print(i,'missing_check');continue
 r,v,o,c=[json.loads(x.read_text()) for x in paths];on=next(x['candidate'] for x in o['records'] if abs(x['gamma']-.0203)<1e-8 and x['initial_type']=='previous_pwa_cycle');onval=f(on['period']) if on['status']=='cycle' else on['status'];branches=p/f'fold-branches-{i}.json';ok=False
 if branches.exists():
  b=json.loads(branches.read_text());q=[z for z in b.get('branches',[]) if z.get('closure',1)<1e-6 and z.get('min_C_nodes',-1)>=-1e-8 and z.get('min_I_nodes',-1)>=-1e-8 and z.get('min_h_nodes',-1)>=-1e-8 and z.get('max_h_nodes',2)<=1+1e-8]
  for g in {z['gamma'] for z in q}:
   group=[z for z in q if z['gamma']==g];ok=ok or (any(z.get('spectral_radius',2)<1-1e-5 for z in group) and any(z.get('spectral_radius',0)>1+1e-5 for z in group))
 m=v['conditional_MAE'];leads=c['region_max_real_eig'];print(i,f(r['best_cost']),f"{v['detected_cycles']}/{v['total']}",f(m['Cmax']),f(m['Imax']),onval,sci(c.get('closure')),sci(c.get('fold_determinant')),f(leads[0]),f(leads[1]),ok,r['optimizer_success'],r.get('start_perturbed',False))
