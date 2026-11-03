#!/usr/bin/env python3
import sys,json
from pathlib import Path
p=Path(sys.argv[1]);print('seed objective cycles/total Cmax_MAE Imax_MAE period_MAE onset_0203_previous_PWA fold_closure fold_phase fold_physical lower1_eig lower2_eig solver_success perturbed')
def f(x):return None if x is None else round(x,5)
for path in sorted(p.glob('candidate-*.json'),key=lambda x:int(x.stem.split('-')[-1])):
 c=json.loads(path.read_text());i=path.stem.split('-')[-1];paths=[p/f'{z}-{i}.json' for z in ('validation','onset','fold-check')]
 if not all(q.exists() for q in paths):print(i,'missing_check');continue
 v,o,fold=[json.loads(q.read_text()) for q in paths];on=next(x['candidate'] for x in o['records'] if abs(x['gamma']-.0203)<1e-8 and x['initial_type']=='previous_pwa_cycle');onval=f(on['period']) if on['status']=='cycle' else on['status'];m=v['conditional_MAE'];print(i,f(c['objective']),f"{v['detected_cycles']}/{v['total']}",f(m['Cmax']),f(m['Imax']),f(m['period']),onval,f(fold.get('closure')),f(fold.get('phase_Cdot')),fold.get('physical_at_checked_points'),*[f(x) for x in c['lower_zone_leading_eigenvalues']],c['optimizer_success'],c['start_perturbed'])
