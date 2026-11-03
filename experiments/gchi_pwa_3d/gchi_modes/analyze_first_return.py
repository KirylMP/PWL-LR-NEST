#!/usr/bin/env python3
"""Assess first-return derivative and local elimination at fixed input.

Analysis, not a fold fit: 3 residuals and 21 anchor-preserving coefficients.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import null_space,qr
from scipy.optimize import least_squares
from fit_modes import design
from poincare_fm import evaluate,section_point_from_peak,NoReturn
from poincare_fm_exact import evaluate_exact

def clean(r):
 return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--cycle',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--section',type=float,default=.2);ap.add_argument('--solve',action='store_true');ap.add_argument('--exact',action='store_true',help='use event-bracketed matrix exponentials');ap.add_argument('--max-nfev',type=int,default=25);a=ap.parse_args();d=json.loads(a.candidate.read_text());o=json.loads(a.cycle.read_text());o=o[0] if isinstance(o,list) else o;co=np.array(d['coefficients']);kn=d['knots'];g=o['gamma'];anc=np.array([q['state'] for q in d['fixed_equilibria']]);ag=np.array([q['gamma'] for q in d['fixed_equilibria']]);D=design(anc,ag,kn);N=null_space(D);shape=(N.shape[1],3);v0=(N.T@co).ravel();u=section_point_from_peak(co,kn,g,o['state'],a.section)
 def mapping(v,u):return (evaluate_exact if a.exact else evaluate)(N@v.reshape(shape),kn,g,u,c_section=a.section)
 def residual(v,u):
  z=mapping(v,u);return np.r_[z['return_state']-u,z['fold_residual']]
 base=mapping(v0,u);adaptive=evaluate(co,kn,g,u,c_section=a.section);r0=np.r_[base['return_state']-u,base['fold_residual']];delta=2e-5;fd=np.zeros((2,2))
 for j in range(2):
  h=np.eye(2)[j]*delta;fd[:,j]=(mapping(v0,u+h)['return_state']-mapping(v0,u-h)['return_state'])/(2*delta)
 print('P derivative error',np.max(abs(fd-base['DP'])),'base residual',r0,flush=True)
 J=np.empty((3,len(v0)));step=1e-4
 for j in range(len(v0)):
  dv=np.eye(len(v0))[j]*step
  try:J[:,j]=(residual(v0+dv,u)-residual(v0-dv,u))/(2*step)
  except NoReturn as ex:raise RuntimeError(f'cannot differentiate coefficient {j}: {ex}')
  if j%5==0:print('coefficient derivative',j,'/',len(v0),flush=True)
 # Row/column scaling avoids choosing pivots solely due to mixed units.
 rs=np.array([.1,.5,1.]);cs=np.maximum(.3,np.minimum(abs(v0),3.));Js=(J/rs[:,None])*cs[None,:];_,_,permutation=qr(Js,pivoting=True,mode='economic');pivot=permutation[:3];sing=np.linalg.svd(Js[:,pivot],compute_uv=False)
 out={'method':'matrix_exponential' if a.exact else 'adaptive_ivp','adaptive_difference':{'return':float(np.max(abs(base['return_state']-adaptive['return_state']))),'time':float(abs(base['return_time']-adaptive['return_time'])),'DP':float(np.max(abs(base['DP']-adaptive['DP'])))},'gamma':float(g),'section_C':a.section,'section_state':u.tolist(),'base_return':clean(base),'closure':r0[:2].tolist(),'fold_residual':float(r0[2]),'finite_difference_DP':fd.tolist(),'DP_max_abs_error':float(np.max(abs(fd-base['DP']))),'anchor_matrix_rank':int(np.linalg.matrix_rank(D)),'anchor_null_basis':N.tolist(),'field_coefficients_before':int(co.size),'field_coefficients_after_anchors':int(v0.size),'free_return_coordinates':2,'eliminated_coefficients':3,'remaining_local_free_dimensions':int(v0.size+2-3),'scaled_pivot_columns':pivot.tolist(),'scaled_pivot_singular_values':sing.tolist(),'scaled_pivot_condition':float(sing[0]/sing[-1]),'coefficient_jacobian_rank':int(np.linalg.matrix_rank(Js,tol=1e-7)),'scaled_coefficient_jacobian':Js.tolist(),'limitations':'Local derivative at the tested point only; first-return section/transversality and numerical crossing checks are required. Rank alone does not prove a nondegenerate fold or global chart.'}
 if a.solve:
  basep=v0[pivot].copy();n=0
  def fun(z):
   nonlocal n
   n+=1;vv=v0.copy();vv[pivot]=z
   try:r=residual(vv,u)/rs
   except NoReturn:return np.full(3,100.)+0.01*(z-basep)
   if n%10==0:print('inner evaluations',n,'scaled residual',round(float(np.linalg.norm(r)),6),flush=True)
   return r
  sol=least_squares(fun,basep,bounds=(basep-2,basep+2),max_nfev=a.max_nfev,diff_step=1e-4,ftol=1e-7,xtol=1e-7,gtol=1e-7);vv=v0.copy();vv[pivot]=sol.x;cofit=N@vv.reshape(shape)
  try:
   point=mapping(vv,u);res=np.r_[point['return_state']-u,point['fold_residual']];region=[];m=cofit[1:4,:].T.copy()
   for j in range(4):
    if j:m[:,0]+=cofit[6+j]
    region.append(float(max(np.linalg.eigvals(m).real)))
   out['inner_attempt']={'solver_success':bool(sol.success),'equations_satisfied':bool(np.linalg.norm(res)<1e-6),'status':str(sol.message),'nfev':int(sol.nfev),'residual':res.tolist(),'scaled_norm':float(np.linalg.norm(res/rs)),'field_coefficients':cofit.tolist(),'region_max_real_eig':region,'anchor_residual':float(np.max(abs(D@cofit))),'return':clean(point),'note':'Attempt solves the 3 equations with section coordinates and 18 other coefficients fixed. Physical admissibility, global cycle first return and bifurcation nondegeneracy are not guaranteed.'}
  except NoReturn as ex:out['inner_attempt']={'status':str(ex),'success':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('base_return','inner_attempt')},indent=2),flush=True)
 if 'inner_attempt' in out:print('inner',str(out['inner_attempt'])[:1200],flush=True)
if __name__=='__main__':main()
