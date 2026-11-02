#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
print('case seed status hold_cycles period_MAPE% amplitude_MAPE% loss spectrum_margin_pass')
for f in sorted(a.directory.rglob('result.json')):
 d=json.loads(f.read_text()); hold=[r for r in d['validation'] if r['split']!='train'];good=[r for r in hold if r['candidate']['status']=='cycle_detected'];
 def avg(kind):
  v=[]
  for x in good:
   q=x['candidate'];t=x['target']
   if kind=='period':v.append(abs(q['period']/t['period']-1))
   else:v.append(abs((q['Cmax']-q['Cmin'])/(t['Cmax']-t['Cmin'])-1))
  return f'{100*np.mean(v):.2f}' if v else 'NA'
 j=[np.array(z) for z in d['jacobians']];tr=[np.trace(z) for z in j];det=[np.linalg.det(z) for z in j]
 ok=tr[0]<=-.01 and tr[1]>=.01 and tr[2]<=-.01 and min(det)>=.005 and tr[1]**2-4*det[1]<=-.005 and tr[2]**2-4*det[2]<=-.005
 print(d['case'],d['seed'],d['status'],f'{len(good)}/{len(hold)}',avg('period'),avg('amplitude'),f"{d['training_objective']:.3g}",ok, str(f))
