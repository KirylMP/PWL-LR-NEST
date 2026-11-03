#!/usr/bin/env python3
"""Small local refinement on the implicit first-return fold manifold.

At each outer evaluation three pivot coefficients are solved from the two
fixed-point equations and determinant condition. Reduced objectives are soft
proxies and cannot replace independent cycle/fold validation.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import root,least_squares
from fit_modes import design
from explore_modes import P,field
from poincare_fm import NoReturn
from poincare_fm_exact import evaluate_exact

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--fold',type=Path,required=True);ap.add_argument('--chart',type=Path,required=True);ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--scan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-nfev',type=int,default=3);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--reference-fold',type=Path,default=Path(__file__).parent/'analysis/fm-constrained-01/reference-fold-candidate.json');a=ap.parse_args();d=json.loads(a.candidate.read_text());f=json.loads(a.fold.read_text());ch=json.loads(a.chart.read_text());ref=json.loads(a.reference.read_text());scan=json.loads(a.scan.read_text());kn=np.array(d['knots']);gfold=f['gamma'];sec=f['section_C'];u0=np.array(f['section_state']);anchors=np.array([x['state'] for x in d['fixed_equilibria']]);ag=np.array([x['gamma'] for x in d['fixed_equilibria']]);D=design(anchors,ag,kn);N=np.asarray(ch['anchor_null_basis']);
 if N.shape!=(len(kn)+7,len(kn)+7-3) or np.max(np.abs(D@N))>1e-10 or np.max(np.abs(N.T@N-np.eye(N.shape[1])))>1e-10:raise ValueError('saved anchor null-space basis incompatible with this model')
 shape=(N.shape[1],3);v0=(N.T@np.array(f['coefficients'])).ravel();pivot=np.array(ch['scaled_pivot_columns']);outer=np.array([j for j in range(len(v0)) if j not in pivot]);x0=np.zeros(len(outer)+2);route=[e['threshold_index'] for e in f['return']['switches']];near=[dict(x['result'],gamma=x['gamma']) for x in scan['reference_low'] if x['result']['status']=='cycle'];good=[x for x in ref['cycles'] if x['status']=='cycle'];targets=[]
 for target in [.0203,.021,.07198847153274834,.8560920254491522]:
  q=min(near+good,key=lambda r:abs(r['gamma']-target));g=q['gamma']
  def ode(t,y):return field(t,y,g,P['FM'])
  def event(t,y):return y[0]-sec
  event.direction=1;event.terminal=True
  s=solve_ivp(ode,(0,q['period']),q['peak'],events=event,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.1)
  if not len(s.t_events[0]):raise RuntimeError('reference section crossing unavailable')
  targets.append((q,s.y_events[0][0,1:]))
 print('training gammas',*[r[0]['gamma'] for r in targets],'outer dimensions',len(x0),flush=True);calls=0;best=[np.inf,None,None];ref_fold=json.loads(a.reference_fold.read_text())
 # Residual dimension: 5 fold observables, 4*8 trajectory terms, 4 stability terms, regularization 20.
 nres=5+len(targets)*8+4+len(x0)
 def run(x,record=False):
  nonlocal calls,best
  calls+=1;vv=v0.copy();vv[outer]+=x[:-2];uu=u0+x[-2:]
  def adjust(y):
   v=vv.copy();v[pivot]=y
   try:
    z=evaluate_exact(N@v.reshape(shape),kn,gfold,uu,c_section=sec)
    if [e['threshold_index'] for e in z['switches']]!=route or min(abs(e['speed']) for e in z['switches'])<1e-4:return np.full(3,100.)
    return np.r_[(z['return_state']-uu)/np.array([.1,.5]),z['fold_residual']]
   except (NoReturn,ValueError,OverflowError):return np.full(3,100.)
  sol=root(adjust,v0[pivot],method='hybr',tol=1e-8,options={'eps':1e-7,'maxfev':100});vv[pivot]=sol.x;res=adjust(sol.x)
  if np.linalg.norm(res)>1e-6:return np.full(nres,50.)+.001*np.linalg.norm(x)
  co=N@vv.reshape(shape)
  try:fr=evaluate_exact(co,kn,gfold,uu,c_section=sec)
  except NoReturn:return np.full(nres,50.)+.001*np.linalg.norm(x)
  scale=np.array([.05,.2,.5,.5,2.]);fvals=np.array([fr['min_C_nodes'],fr['max_C_nodes'],fr['min_I_nodes'],fr['max_I_nodes'],fr['return_time']]);ft=np.array([ref_fold['Cmin'],ref_fold['Cmax'],ref_fold['Imin'],ref_fold['Imax'],ref_fold['period']]);residual=list((fvals-ft)/scale);physical=fr['min_C_nodes']>=0 and fr['min_I_nodes']>=0 and fr['min_h_nodes']>=0 and fr['max_h_nodes']<=1
  for q,ui in targets:
   try:
    z=evaluate_exact(co,kn,q['gamma'],ui,c_section=sec);P=z['return_state'];residual.extend([(P[0]-ui[0])/.15,(P[1]-ui[1])/.5,2*(z['return_time']-q['period'])/q['period'],(z['min_C_nodes']-q['Cmin'])/.06,(z['max_C_nodes']-q['Cmax'])/.3,(z['min_I_nodes']-q['Imin'])/.5,(z['max_I_nodes']-q['Imax'])/.7,4*max(-z['min_I_nodes'],0)/.1+2*max(-z['min_C_nodes'],0)/.1])
   except (NoReturn,ValueError,OverflowError):residual.extend([10.]*8)
  A=co[1:4].T.copy();lead=[]
  for j in range(4):
   if j:A[:,0]+=co[6+j]
   lead.append(float(max(np.linalg.eigvals(A).real)))
  residual.extend((np.maximum(.001-np.array([-1,-1,1,-1])*np.array(lead),0)/.1).tolist());residual.extend((.1*x).tolist());residual=np.clip(np.nan_to_num(residual,nan=1e3,posinf=1e3,neginf=-1e3),-1e3,1e3);cost=float(residual@residual);feasible=physical and max(lead[:2])<-.001 and lead[2]>.001 and lead[3]<-.001 and np.linalg.norm(res)<1e-6
  if feasible and cost<best[0]:best=[cost,x.copy(),{'coefficients':co.tolist(),'section_state':uu.tolist(),'gamma':gfold,'section_C':sec,'return_time':fr['return_time'],'fold_equation_residual_scaled':res.tolist(),'fold_equation_residual':np.r_[fr['return_state']-uu,fr['fold_residual']].tolist(),'leading_eig':lead,'fold_extrema_sampled':fvals.tolist()}]
  if calls%25==1:print('calls',calls,'cost',round(cost,5),'best',round(best[0],5),'feasible',feasible,flush=True)
  return residual
 def jac(x):
  h=1e-4;J=np.empty((nres,len(x)))
  for j in range(len(x)):
   e=np.eye(len(x))[j]*h;J[:,j]=(run(x+e)-run(x-e))/(2*h)
  return J
 initial=run(x0);perturbed=False
 if a.seed:
  rng=np.random.default_rng(a.seed);trial=rng.normal(0,.001,size=x0.shape);run(trial)
  if best[1] is not None and np.array_equal(best[1],trial):x0=trial;perturbed=True
 sol=least_squares(run,x0,jac=jac,bounds=(np.r_[np.full(len(outer),-.05),[-.02,-.05]],np.r_[np.full(len(outer),.05),[.02,.05]]),max_nfev=a.max_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5,verbose=1)
 if best[1] is None:raise RuntimeError('no feasible fold-preserving evaluation')
 result={'initial_cost':float(initial@initial),'best_cost':best[0],'outer_dimension':len(x0),'training_gammas':[float(q['gamma']) for q, _ in targets],'eliminated_coefficient_indices':pivot.tolist(),'calls':calls,'seed':a.seed,'start_perturbed':perturbed,'outer_nfev':int(sol.nfev),'first_order_optimality':float(sol.optimality),'status':sol.message,'optimizer_success':bool(sol.success),'solution':best[2],'scope':'Local finite-difference fit with implicit first-return shooting. Reference-section returns are surrogates, soft physical penalties; independent orbit continuation and validation required.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print('DONE',result['initial_cost'],result['best_cost'],sol.message,flush=True)
if __name__=='__main__':main()
