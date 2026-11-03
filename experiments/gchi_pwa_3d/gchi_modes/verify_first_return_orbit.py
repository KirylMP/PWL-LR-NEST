#!/usr/bin/env python3
"""Independent adaptive verification of a first-return fold orbit."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_modes import design

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--fold',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.candidate.read_text());z=json.loads(a.fold.read_text());co=np.array(z['coefficients']);kn=d['knots'];g=z['gamma'];C=z['section_C'];y=np.r_[C,z['section_state']];T=z['return']['return_time']
 def field(t,x):return design(x,g,kn)[0]@co
 def rhs(t,w):
  x=w[:3];A=co[1:4].T.copy()
  for j,k in enumerate(kn):
   if x[0]>k:A[:,0]+=co[7+j]
  return np.r_[field(t,x),(A@w[3:].reshape(3,3)).ravel()]
 def section(t,w):return w[0]-C
 section.direction=1;section.terminal=False
 s=solve_ivp(rhs,(0,T+1e-4),np.r_[y,np.eye(3).ravel()],events=section,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.01,dense_output=True)
 if not s.success:raise RuntimeError(s.message)
 events=s.t_events[0].tolist();w=s.sol(T);M=w[3:].reshape(3,3);mu=np.linalg.eigvals(M);data=s.sol(np.linspace(0,T,10000))[:3];out={'gamma':g,'period':T,'section_crossings':events,'closure':float(np.linalg.norm(w[:3]-y)),'multipliers':[[float(z.real),float(z.imag)] for z in mu],'sampled_min_C':float(min(data[0])),'sampled_max_C':float(max(data[0])),'sampled_min_I':float(min(data[2])),'sampled_max_I':float(max(data[2])),'sampled_min_h':float(min(data[1])),'sampled_max_h':float(max(data[1])),'scope':'Adaptive variational integration with finite sampling; C-switching events are not used to split integration. No formal proof of positivity, non-grazing or nondegeneracy.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
