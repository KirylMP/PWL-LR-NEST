#!/usr/bin/env python3
"""Finite continuation-by-integration of an attracting FM PWA response."""
import argparse,json
from pathlib import Path
import numpy as np
from validate_modes import simulate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--initial',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=700);a=ap.parse_args();c=json.loads(a.candidate.read_text());co=np.array(c['coefficients']);kn=np.array(c['knots']);y=json.loads(a.initial.read_text());rows=[]
 for g in [21.202553865109465,30.,50.,100.,300.,1000.]:
  initial=y;z=simulate(g,y,co,kn,a.duration);rows.append({'gamma':g,'initial':initial,'result':z});print(g,z['status'],flush=True)
  if z['status']!='cycle':break
  y=z['end']
 out={'records':rows,'scope':'One branch sampled by forward integration, not pseudo-arclength continuation; folds in gaps and other branches may be missed.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
