#!/usr/bin/env python3
import sys,json
from pathlib import Path
p=Path(sys.argv[1]);print('seed objective cycles/total lower1_eig lower2_eig Cmax_MAE period_MAE solver_success perturbed')
for f in sorted(p.glob('candidate-*.json'),key=lambda x:int(x.stem.split('-')[-1])):
 d=json.loads(f.read_text());vf=p/('validation-'+f.stem.split('-')[-1]+'.json')
 if not vf.exists():print(d['seed'],'missing validation');continue
 v=json.loads(vf.read_text());m=v['conditional_MAE'];fmt=lambda x:None if x is None else round(x,4)
 print(d['seed'],round(d['objective'],5),f"{v['detected_cycles']}/{v['total']}",*[round(x,5) for x in d['lower_zone_leading_eigenvalues']],fmt(m['Cmax']),fmt(m['period']),d['optimizer_success'],d.get('start_perturbed',False))
