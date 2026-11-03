#!/usr/bin/env python3
"""Compare FM near-onset candidate responses from two initial-state families."""
import argparse,json
from pathlib import Path
import numpy as np
from validate_modes import simulate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--scan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=1600);a=ap.parse_args();c=json.loads(a.candidate.read_text());s=json.loads(a.scan.read_text());co=np.array(c['coefficients']);kn=np.array(c['knots']);previous=next(r['result']['end'] for r in s['PWA_low'] if abs(r['gamma']-.0203)<1e-8);out=[]
 for g in [.0203,.021]:
  ref=next(r['result'] for r in s['reference_low'] if abs(r['gamma']-g)<1e-8)
  for label,y in [('reference_peak',ref['peak']),('previous_pwa_cycle',previous)]:
   z=simulate(g,y,co,kn,a.duration);out.append({'gamma':g,'initial_type':label,'initial':y,'reference':{k:ref[k] for k in ('period','Cmin','Cmax','Imin','Imax')},'candidate':z});print('gamma',g,label,z['status'],z.get('period'),flush=True)
 result={'duration':a.duration,'records':out,'scope':'Finite adaptive integrations from two initial states; detection or non-detection does not prove absence of other cycles or physical invariance.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
