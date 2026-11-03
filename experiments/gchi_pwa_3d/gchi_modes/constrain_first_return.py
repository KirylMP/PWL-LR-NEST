#!/usr/bin/env python3
"""Local constrained shooting: fixed anchor equilibria, first return and fold."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize
from fit_modes import design
from poincare_fm import NoReturn,evaluate
from poincare_fm_exact import evaluate_exact

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--pivots',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--maxiter',type=int,default=150);ap.add_argument('--start-stable',action='store_true');a=ap.parse_args();d=json.loads(a.candidate.read_text());an=json.loads(a.analysis.read_text());rows=json.loads(a.pivots.read_text())['results'];cands=[r for r in rows if r.get('physical_at_scan_nodes') and (r.get('required_region_stability') if a.start_stable else r.get('equations_satisfied'))];
 if not cands:raise RuntimeError('no initial algebraic shooting solution')
 initial=min(cands,key=lambda z:z['norm']);kn=d['knots'];g=an['gamma'];u=np.array(an['section_state']);sec=an['section_C'];anchors=np.array([q['state'] for q in d['fixed_equilibria']]);ag=np.array([q['gamma'] for q in d['fixed_equilibria']]);D=design(anchors,ag,kn);N=null_space(D);shape=(N.shape[1],3);vbase=(N.T@np.array(d['coefficients'])).ravel();vstart=(N.T@np.array(initial['coefficients'])).ravel();cs=np.maximum(.3,np.minimum(abs(vbase),3));cached=[None,None];calls=0;t0=time.monotonic()
 def state(v):
  nonlocal calls
  if cached[0] is not None and np.array_equal(v,cached[0]):return cached[1]
  calls+=1;co=N@v.reshape(shape)
  try:
   ret=evaluate_exact(co,kn,g,u,c_section=sec,scan_step=.05);res=np.r_[ret['return_state']-u,ret['fold_residual']];A=co[1:4].T.copy();lead=[]
   for j in range(4):
    if j:A[:,0]+=co[6+j]
    lead.append(float(max(np.linalg.eigvals(A).real)))
   cross=min(abs(q['speed']) for q in ret['switches']) if ret['switches'] else 0.
   # Stable lower+upper regions, unstable middle; sample-domain and transversal margin.
   ineq=np.r_[-.001-lead[0],-.001-lead[1],lead[2]-.001,-.001-lead[3],ret['min_C_nodes']-1e-7,ret['min_I_nodes']-1e-7,ret['min_h_nodes'],1-ret['max_h_nodes'],ret['max_C_nodes']-ret['min_C_nodes']-.1,cross-.0001,ret['crossing_speed']-.0001,ret['return_time']-3,30-ret['return_time']]
  except (NoReturn,ValueError,OverflowError):ret=None;res=np.ones(3)*100;ineq=np.ones(13)*-100;lead=[None]*4
  result=(res,ineq,ret,lead)
  cached[:]=[v.copy(),result]
  if calls%100==1:print('calls',calls,'res',round(float(np.linalg.norm(res)),6),'minineq',round(float(min(ineq)),6),'elapsed',round(time.monotonic()-t0,1),flush=True)
  return result
 def objective(v):return float(np.sum(((v-vbase)/cs)**2))
 def eq(v):return state(v)[0]/np.array([.1,.5,1])
 def ineq(v):return state(v)[1]
 bounds=list(zip(vbase-8,vbase+8));sol=minimize(objective,vstart,method='SLSQP',bounds=bounds,constraints=[{'type':'eq','fun':eq},{'type':'ineq','fun':ineq}],options={'maxiter':a.maxiter,'ftol':1e-9,'eps':1e-5,'disp':True});res,ineqs,ret,lead=state(sol.x);co=N@sol.x.reshape(shape);valid=ret is not None and np.linalg.norm(res)<1e-6 and min(ineqs)>=-1e-7;out={'initial_pivot':initial['pivot'],'gamma':g,'section_C':sec,'section_state':u.tolist(),'solver_success':bool(sol.success),'status':str(sol.message),'iterations':int(sol.nit),'calls':calls,'equations_satisfied':bool(np.linalg.norm(res)<1e-6),'inequalities_satisfied':bool(min(ineqs)>=-1e-7),'numerically_valid_candidate':bool(valid),'residual':res.tolist(),'inequalities':ineqs.tolist(),'objective':float(objective(sol.x)),'region_max_real_eig':lead,'anchor_residual':float(np.max(abs(D@co))),'coefficients':co.tolist(),'note':'Numerical first-return solution subject to finite scan, physical and stability checks; no assertion of nondegenerate fold without independent verification.'}
 if ret is not None:
  out['return']={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in ret.items()}
  if valid:
   try:
    fine=evaluate_exact(co,kn,g,u,c_section=sec,scan_step=.025);adaptive=evaluate(co,kn,g,u,c_section=sec,max_step=.025,rtol=2e-10,atol=2e-12);out['verification']={'fine_residual':np.r_[fine['return_state']-u,fine['fold_residual']].tolist(),'adaptive_residual':np.r_[adaptive['return_state']-u,adaptive['fold_residual']].tolist(),'DP_exact_vs_adaptive_max_abs':float(np.max(abs(fine['DP']-adaptive['DP']))),'first_return_time_exact':fine['return_time'],'first_return_time_adaptive':adaptive['return_time']}
   except NoReturn as e:out['verification']={'error':str(e)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print('DONE',sol.message,'valid',valid,'res',res,'ineq_min',min(ineqs),flush=True)
if __name__=='__main__':main()
