#!/usr/bin/env python3
import sys,json
from pathlib import Path
p=Path(sys.argv[1]);print('seed objective cycles/total Cmin_MAE Cmax_MAE Imin_MAE Imax_MAE period_MAE')
for file in sorted(p.glob('candidate-*.json'),key=lambda x:int(x.stem.split('-')[-1])):
 d=json.loads(file.read_text());vpath=p/('validation-'+file.stem.split('-')[-1]+'.json')
 if not vpath.exists():print(d.get('seed'),d['objective'],'missing validation');continue
 v=json.loads(vpath.read_text());mae=v['conditional_MAE'];print(d.get('seed'),round(d['objective'],6),f"{v['detected_cycles']}/{v['total']}",*[None if mae[k] is None else round(mae[k],5) for k in ('Cmin','Cmax','Imin','Imax','period')])
