#!/usr/bin/env python3
"""Attempt local movement along first-return fold equations toward stable cells."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares,minimize
from fit_modes import design
from poincare_fm import NoReturn,evaluate
from poincare_fm_exact import evaluate_exact

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--pivots',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--iterations',type=int,default=15);a=ap.parse_args();d=json.loads(a.candidate.read_text());an=json.loads(a.analysis.read_text());rr=json.loads(a.pivots.read_text())['results'];r=next(z for z in rr if z.get('equations_satisfied') and z.get('physical_at_scan_nodes'));pivot=np.array(r['pivot']);kn=np.array(d['knots']);g=an['gamma'];u=np.array(an['section_state']);sec=an['section_C'];anchors=np.array([q['state'] for q in d['fixed_equilibria']]);ag=np.array([q['gamma'] for q in d['fixed_equilibria']]);D=design(anchors,ag,kn);N=null_space(D);shape=(N.shape[1],3);v=(N.T@np.array(r['coefficients'])).ravel();outer=np.array([i for i in range(len(v)) if i not in pivot]);scale=np.array([.1,.5,1.]);history=[]
 def mapping(v,gamma=g,uu=u,step=.05):return evaluate_exact(N@v.reshape(shape),kn,gamma,uu,c_section=sec,scan_step=step)
 def residual(v):
  z=mapping(v);return np.r_[z['return_state']-u,z['fold_residual']]
 def eigs(v):
  co=N@v.reshape(shape);A=co[1:4].T.copy();out=[]
  for j in range(4):
   if j:A[:,0]+=co[6+j]
   out.append(float(max(np.linalg.eigvals(A).real)))
  return np.array(out)
 for it in range(a.iterations):
  z=mapping(v);r0=residual(v);e=eigs(v);physical=z['min_C_nodes']>=0 and z['min_I_nodes']>=0 and z['min_h_nodes']>=0 and z['max_h_nodes']<=1;history.append({'iteration':it,'residual':r0.tolist(),'leading_eig':e.tolist(),'physical_at_nodes':bool(physical)});print('iteration',it,'eig',np.round(e,6),'res',np.linalg.norm(r0),'physical',physical,flush=True)
  if max(e[0:2])<-.001 and e[2]>.001 and e[3]<-.001 and physical:break
  h=2e-5;J=np.empty((3,len(v)));E=np.empty((4,len(v)))
  for j in range(len(v)):
   dv=np.eye(len(v))[j]*h
   try:J[:,j]=(residual(v+dv)-residual(v-dv))/(2*h)
   except NoReturn:raise RuntimeError(f'Jacobian unavailable at coordinate {j}')
   E[:,j]=(eigs(v+dv)-eigs(v-dv))/(2*h)
  Jp=J[:,pivot]
  if np.linalg.cond(Jp)>1e9:raise RuntimeError('implicit pivot Jacobian ill-conditioned')
  tangent=np.zeros((len(v),len(outer)));tangent[outer,np.arange(len(outer))]=1;tangent[pivot,:]=-np.linalg.solve(Jp,J[:,outer]);G=E@tangent
  # Linearized stability target on the tangent space; regularize size and limit steps.
  def obj(x):return .5*float(x@x)
  def con(x):
   z=e+G@x;return np.array([-.003-z[0],-.003-z[1],z[2]-.003,-.003-z[3]])
  plan=minimize(obj,np.zeros(len(outer)),jac=lambda x:x,method='SLSQP',bounds=[(-.1,.1)]*len(outer),constraints=[{'type':'ineq','fun':con}],options={'maxiter':100,'ftol':1e-10})
  if not plan.success:print('linearized step',plan.message,flush=True)
  accepted=False
  for factor in [1,.5,.25,.125,.0625]:
   vt=v.copy();vt[outer]+=factor*plan.x;p0=vt[pivot].copy()
   def f(z):
    trial=vt.copy();trial[pivot]=z
    try:return residual(trial)/scale
    except NoReturn:return np.ones(3)*100+.001*(z-p0)
   correction=least_squares(f,p0,max_nfev=80,diff_step=2e-5,xtol=1e-10,ftol=1e-10,gtol=1e-10);vt[pivot]=correction.x
   try:
    R=residual(vt);et=eigs(vt);zt=mapping(vt);good=np.linalg.norm(R)<1e-6 and max(et[:2])<max(e[:2])-1e-5 and et[2]>0 and et[3]<0 and zt['min_C_nodes']>=0 and zt['min_I_nodes']>=0 and zt['min_h_nodes']>=0 and zt['max_h_nodes']<=1
   except NoReturn:good=False
   if good:v=vt;accepted=True;break
  if not accepted:print('STOP: no admissible projected stability step',flush=True);break
 co=N@v.reshape(shape);z=mapping(v);R=residual(v);e=eigs(v);valid=np.linalg.norm(R)<1e-6 and max(e[:2])<-.001 and e[2]>.001 and e[3]<-.001 and z['min_C_nodes']>=0 and z['min_I_nodes']>=0 and z['min_h_nodes']>=0 and z['max_h_nodes']<=1
 out={'gamma':g,'section_C':sec,'section_state':u.tolist(),'history':history,'residual':R.tolist(),'leading_eig':e.tolist(),'anchor_residual':float(np.max(abs(D@co))),'coefficients':co.tolist(),'return':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in z.items()},'candidate_satisfies_checks':bool(valid),'note':'Local projected continuation of algebraic fold conditions; scan-node domain checks and solver tolerance do not prove global physical invariance or nondegeneracy.'}
 if valid:
  fine=evaluate_exact(co,kn,g,u,c_section=sec,scan_step=.025);adapt=evaluate(co,kn,g,u,c_section=sec,max_step=.025,rtol=2e-10,atol=2e-12);out['verification']={'fine_residual':np.r_[fine['return_state']-u,fine['fold_residual']].tolist(),'adaptive_residual':np.r_[adapt['return_state']-u,adapt['fold_residual']].tolist(),'DP_error':float(np.max(abs(fine['DP']-adapt['DP']))),'return_time_exact':fine['return_time'],'return_time_adaptive':adapt['return_time']}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print('DONE valid',valid,'eigs',e,'res',R,flush=True)
if __name__=='__main__':main()
