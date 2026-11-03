#!/usr/bin/env python3
"""Sample PWA periodic-orbit multipliers, using direct integration and peak returns."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_modes import design

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--initial-json',type=Path,required=True,help='JSON array [C,h,I]');ap.add_argument('--gammas',type=float,nargs='+',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.candidate.read_text());co=np.array(d['coefficients']);kn=np.array(d['knots']);y=json.loads(a.initial_json.read_text());out=[]
 for g in a.gammas:
  def f(t,x):return design(x,g,kn)[0]@co
  def peak(t,x):return f(t,x)[0]
  peak.direction=-1
  s=solve_ivp(f,(0,500),y,events=peak,rtol=2e-9,atol=2e-11,max_step=.12,dense_output=True)
  t=s.t_events[0]
  if not s.success or len(t)<4:raise RuntimeError(f'No periodic seed at gamma {g}')
  T=t[-1]-t[-2];y=s.y_events[0][-1]
  if np.linalg.norm(s.sol(t[-1])-s.sol(t[-2]))>1e-5:raise RuntimeError(f'Peak return not converged at gamma {g}')
  def rhs(t,u):
   x=u[:3];A=co[1:4].T.copy()
   for j,k in enumerate(kn):
    if x[0]>k:A[:,0]+=co[7+j]
   return np.r_[f(t,x),(A@u[3:].reshape(3,3)).ravel()]
  z=solve_ivp(rhs,(0,T),np.r_[y,np.eye(3).ravel()],rtol=1e-9,atol=1e-11,method='DOP853',max_step=.04)
  mu=sorted(np.linalg.eigvals(z.y[3:,-1].reshape(3,3)),key=lambda q:abs(q-1));record={'gamma':g,'state':y.tolist(),'period':float(T),'multipliers':[[float(q.real),float(q.imag)] for q in mu],'closure':float(np.linalg.norm(z.y[:3,-1]-y)),'neutral_multiplier_error':float(abs(mu[0]-1)),'note':'Variational integration for continuous PWA field; switching events and grazing are not analytically resolved.'};out.append(record);print('gamma',g,'nontrivial rho',max(abs(mu[1]),abs(mu[2])),'closure',record['closure'],flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
