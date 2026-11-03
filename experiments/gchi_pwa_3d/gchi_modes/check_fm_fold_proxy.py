#!/usr/bin/env python3
"""Adaptive independent evaluation of the fitted FM periodic-orbit fold proxy."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_modes import design

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.candidate.read_text());co=np.array(d['coefficients']);kn=np.array(d['knots']);q=d['fold_proxy'];g=float(q['gamma']);y=np.array(q['state']);T=float(q['period'])
 def field(t,x):return design(x,g,kn)[0]@co
 def rhs(t,u):
  x=u[:3];A=co[1:4,:].T.copy()
  for j,k in enumerate(kn):
   if x[0]>k:A[:,0]+=co[7+j]
  return np.r_[field(t,x),(A@u[3:].reshape(3,3)).ravel()]
 def domain(t,u):return min(u[0]+.05,3-u[0],u[1]+.2,2-u[1],u[2]+.05,5-u[2])
 domain.terminal=True
 s=solve_ivp(rhs,(0,T),np.r_[y,np.eye(3).ravel()],method='DOP853',rtol=1e-9,atol=1e-11,max_step=.035,dense_output=True,events=domain)
 out={'gamma':g,'period':T,'integration_success':bool(s.success and s.t[-1]>=T-1e-8),'t_end':float(s.t[-1]),'start_state':y.tolist(),'min_C_nodes':float(min(s.y[0])),'min_I_nodes':float(min(s.y[2])),'min_h_nodes':float(min(s.y[1])),'max_h_nodes':float(max(s.y[1])),'phase_Cdot':float(field(0,y)[0]),'note':'One unconverged fitted shooting point; not proof of an orbit, fold, nondegeneracy, transversality, or global physical invariance.'}
 if out['integration_success']:
  mu=np.linalg.eigvals(s.y[3:,-1].reshape(3,3));vals=s.sol(np.linspace(0,T,1000))[:3];out.update(closure=float(np.linalg.norm(s.y[:3,-1]-y)),variational_eigenvalues=[[float(x.real),float(x.imag)] for x in mu],Cmin=float(min(vals[0])),Cmax=float(max(vals[0])),Imin=float(min(vals[2])),Imax=float(max(vals[2])),min_C_sampled=float(min(vals[0])),min_I_sampled=float(min(vals[2])),min_h_sampled=float(min(vals[1])),max_h_sampled=float(max(vals[1])))
  out['physical_at_checked_points']=bool(min(s.y[0])>=-1e-8 and min(s.y[2])>=-1e-8 and min(s.y[1])>=-1e-8 and max(s.y[1])<=1+1e-8 and min(vals[0])>=-1e-8 and min(vals[2])>=-1e-8 and min(vals[1])>=-1e-8 and max(vals[1])<=1+1e-8)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
