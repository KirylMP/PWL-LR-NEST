#!/usr/bin/env python3
"""Small directional scans to compare FM candidates near onset and at high glutamate.

Forward integration is not numerical continuation: branches may be missed.
"""
import argparse,json
from pathlib import Path
import numpy as np
from validate_modes import simulate as pwa_simulate
from explore_modes import simulate as ref_simulate,P

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--validation',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.candidate.read_text());v=json.loads(a.validation.read_text());ref=json.loads(a.reference.read_text());co=np.array(d['coefficients']);kn=np.array(d['knots']);records={}
 def go_pwa(values,y):
  out=[]
  for g in values:
   previous=y;z=pwa_simulate(g,y,co,kn,1000);y=z['end'];out.append({'gamma':g,'initial':previous,'result':z});print('PWA',g,z['status'],flush=True)
  return out
 lowstart=next(x['candidate']['end'] for x in v['records'] if .036<x['gamma']<.037);records['PWA_low']=go_pwa([.024,.02336,.0203,.0199],lowstart)
 highstart=next(x['candidate']['end'] for x in v['records'] if x['gamma']==20);records['PWA_high']=go_pwa([20,30,50,100,1000,10000],highstart)
 y=next(x['end'] for x in ref['cycles'] if .0233<x['gamma']<.0234);out=[]
 for g in [.021,.0203,.0199]:
  prior=y;z=ref_simulate(g,P['FM'],y,2000);y=z['end'];out.append({'gamma':g,'initial':prior,'result':z});print('reference',g,z['status'],flush=True)
 records['reference_low']=out;records['scope']='Finite forward integrations from one prior orbit; not an exhaustive basin search or proof of cycle folds, stability multipliers, or global positivity.';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
