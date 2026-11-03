#!/usr/bin/env python3
"""Find PWA equilibrium/threshold intersections on a sampled glutamate range."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from fit_modes import design

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-glu',type=float,default=20);a=ap.parse_args()
 d=json.loads(a.candidate.read_text());co=np.array(d['coefficients']);kn=np.array(d['knots']);cs=[kn[0]/2,(kn[0]+kn[1])/2,(kn[1]+kn[2])/2,kn[2]+.05]
 def matrix(j):
  A=co[1:4].T.copy()
  for k in range(j):A[:,0]+=co[7+k]
  return A
 def equilibrium(g,j):
  y0=np.array([cs[j],.8,.4]);return y0-np.linalg.solve(matrix(j),design(y0,g,kn)[0]@co)
 grid=np.r_[0.,np.geomspace(1e-7,a.max_glu,800)];events=[]
 for j,k in enumerate(kn):
  # Evaluate from one adjacent affine cell; continuity supplies the same boundary value.
  try:values=[equilibrium(g,j)[0]-k for g in grid]
  except np.linalg.LinAlgError:continue
  for x,y,fx,fy in zip(grid[:-1],grid[1:],values[:-1],values[1:]):
   if fx*fy>0:continue
   g=brentq(lambda z:equilibrium(z,j)[0]-k,x,y);z=equilibrium(g,j)
   if min(z[0],z[2]) < -1e-8 or not -1e-8<=z[1]<=1+1e-8:continue
   if events and any(abs(e['gamma']-g)<1e-8 and e['threshold_index']==j for e in events):continue
   events.append({'threshold_index':j,'C_threshold':float(k),'gamma':float(g),'state':z.tolist(),'max_real_below_C':float(max(np.linalg.eigvals(matrix(j)).real)),'max_real_above_C':float(max(np.linalg.eigvals(matrix(j+1)).real))})
 events.sort(key=lambda x:x['gamma']);out={'candidate':str(a.candidate),'events':events,'scope':'Intersection search on sampled gamma grid; may miss closely spaced or tangential roots. Eigenvalues describe affine cells, not verified attractor basins or local HLB type.'};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(events,indent=2))
if __name__=='__main__':main()
