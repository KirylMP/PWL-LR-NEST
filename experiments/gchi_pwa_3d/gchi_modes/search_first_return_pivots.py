#!/usr/bin/env python3
"""Local multi-pivot feasibility probe for the 3 first-return fold equations."""
import argparse,json,itertools
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares
from fit_modes import design
from poincare_fm import NoReturn
from poincare_fm_exact import evaluate_exact

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--count',type=int,default=20);ap.add_argument('--max-nfev',type=int,default=100);a=ap.parse_args();d=json.loads(a.candidate.read_text());z=json.loads(a.analysis.read_text());co=np.array(d['coefficients']);kn=d['knots'];g=z['gamma'];u=np.array(z['section_state']);anc=np.array([q['state'] for q in d['fixed_equilibria']]);ag=np.array([q['gamma'] for q in d['fixed_equilibria']]);D=design(anc,ag,kn);N=null_space(D);shape=(N.shape[1],3);v0=(N.T@co).ravel();Js=np.array(z['scaled_coefficient_jacobian']);scale=np.array([.1,.5,1.]);combos=list(itertools.combinations(range(len(v0)),3));combos.sort(key=lambda p:abs(np.linalg.det(Js[:,p])),reverse=True);selected=[];used=[]
 # Alternate leading coordinates to explore distinct local charts.
 for pivot in combos:
  group=tuple(sorted(j%3 for j in pivot))
  if len(selected)<a.count and (group not in used or len(selected)>=10):selected.append(pivot);used.append(group)
  if len(selected)>=a.count:break
 results=[]
 for pivot in selected:
  pivot=list(pivot);vstart=v0[pivot].copy()
  def calculate(x,step=.05):
   vv=v0.copy();vv[pivot]=x;cc=N@vv.reshape(shape);r=evaluate_exact(cc,kn,g,u,c_section=z['section_C'],scan_step=step);return np.r_[r['return_state']-u,r['fold_residual']],r,cc
  def fun(x):
   try:return calculate(x)[0]/scale
   except (NoReturn,OverflowError,ValueError):return np.full(3,100.)+.001*(x-vstart)
  sol=least_squares(fun,vstart,bounds=(vstart-4,vstart+4),max_nfev=a.max_nfev,diff_step=2e-5,ftol=1e-9,xtol=1e-9,gtol=1e-9)
  try:
   r,ret,cc=calculate(sol.x);regions=[];A=cc[1:4,:].T.copy()
   for j in range(4):
    if j:A[:,0]+=cc[6+j]
    regions.append(float(max(np.linalg.eigvals(A).real)))
   physical=ret['min_C_nodes']>=-1e-8 and ret['min_I_nodes']>=-1e-8 and ret['min_h_nodes']>=-1e-8 and ret['max_h_nodes']<=1+1e-8
   valid_stability=max(regions[:2])<0 and regions[2]>0 and regions[3]<0
   row={'pivot':pivot,'nfev':int(sol.nfev),'solver_success':bool(sol.success),'residual':r.tolist(),'norm':float(np.linalg.norm(r)),'scaled_norm':float(np.linalg.norm(r/scale)),'equations_satisfied':bool(np.linalg.norm(r)<1e-6),'physical_at_scan_nodes':bool(physical),'required_region_stability':bool(valid_stability),'region_max_real_eig':regions,'anchor_residual':float(np.max(np.abs(D@cc))),'coefficients':cc.tolist(),'return_time':ret['return_time'],'section_state':u.tolist(),'crossings':ret['switches'],'fold_residual':ret['fold_residual']}
  except (NoReturn,ValueError) as e:row={'pivot':pivot,'solver_success':False,'error':str(e)}
  results.append(row);print('pivot',pivot,'norm',round(row.get('norm',999),7),'physical',row.get('physical_at_scan_nodes'),'stability',row.get('required_region_stability'),flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'gamma':g,'results':results,'scope':'Local exploratory finite multi-pivot shooting. Fixed section state and all other free coefficients; no guarantee of a global optimum, physical trajectory between samples, or nondegenerate fold.'},indent=2)+'\n')
if __name__=='__main__':main()
