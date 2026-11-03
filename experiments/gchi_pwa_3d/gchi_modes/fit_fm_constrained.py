#!/usr/bin/env python3
"""FM fit with exact anchors, hard lower-cell stability and physical trajectory probes."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import minimize
from fit_modes import design
from explore_modes import P,field

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--start',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--maxiter',type=int,default=80);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--max-step',type=float,default=.035);ap.add_argument('--full-probes',action='store_true',help='also penalize physical violations over full reference periods; slower');args=ap.parse_args()
 d=json.loads(args.start.read_text());ref=json.loads(args.data.read_text());kn=np.array(d['knots']);anchors=np.array([x['state'] for x in d['fixed_equilibria']]);ag=np.array([x['gamma'] for x in d['fixed_equilibria']]);N=null_space(design(anchors,ag,kn));base=np.array(d['coefficients']);v0=(N.T@base).ravel();shape=(N.shape[1],3);p=P['FM']
 good=[x for x in ref['cycles'] if x['status']=='cycle'];idx=np.unique(np.linspace(1,len(good)-2,min(9,len(good)-2),dtype=int));train=[good[i] for i in idx];phases=[];X=[];G=[];Y=[]
 for q in train:
  g=q['gamma'];T=q['period'];s=solve_ivp(field,(0,T),q['peak'],args=(g,p),t_eval=np.linspace(0,T,41),method='DOP853',rtol=1e-9,atol=1e-11,max_step=.15)
  if not s.success:raise RuntimeError('reference integration failed')
  phases.append(s.y[:,::5].T);X.extend(s.y.T[:-1]);G.extend([g]*40);Y.extend([field(0,z,g,p) for z in s.y.T[:-1]])
 probe=[train[j] for j in (0,1,3,6,8)]
 phases=np.array(phases);gg=np.array([q['gamma'] for q in train]);period=np.array([q['period'] for q in train]);ns=max(20,int(np.ceil(max(period)/(8*args.max_step))));dt=np.repeat(period/(8*ns),8);gs=np.repeat(gg,8);X=np.array(X);G=np.array(G);Y=np.array(Y);B=design(X,G,kn)@N;scale=np.array([.4,.15,.4]);calls=0;best=[np.inf,None];start=time.monotonic()
 def matrices(co):
  A=co[1:4].T.copy();out=[A.copy()]
  for i in range(3):A=A.copy();A[:,0]+=co[7+i];out.append(A)
  return out
 def constraints(v):
  co=N@v.reshape(shape);A=matrices(co);leads=np.array([max(np.linalg.eigvals(m).real) for m in A])
  return np.array([-.001-leads[0],-.001-leads[1],leads[2]-.001,-.001-leads[3]])
 def loss(v):
  nonlocal calls,best
  calls+=1;co=N@v.reshape(shape);y=phases[:,:-1,:].reshape(-1,3).copy();mins=y.copy();maxh=y[:,1].copy()
  def f(z):return np.clip(np.nan_to_num(design(np.clip(z,-3,4),gs,kn)@co,nan=100,posinf=100,neginf=-100),-50,50)
  for _ in range(ns):
   k1=f(y);k2=f(y+dt[:,None]*k1/2);k3=f(y+dt[:,None]*k2/2);k4=f(y+dt[:,None]*k3);y=y+dt[:,None]*(k1+2*k2+2*k3+k4)/6;mins=np.minimum(mins,y);maxh=np.maximum(maxh,y[:,1])
  err=(y.reshape(len(gg),8,3)-phases[:,1:,:])/scale;pred=B@v.reshape(shape);ferr=(pred-Y)/np.array([.4,.05,.1]);negI=np.maximum(-mins[:,2],0)/.1;negC=np.maximum(-mins[:,0],0)/.1;badH=(np.maximum(-mins[:,1],0)+np.maximum(maxh-1,0))/.1
  full_pen=np.zeros(40)
  if args.full_probes:
   pg=np.array([q['gamma'] for q in probe]);pt=np.array([q['period'] for q in probe]);pn=int(np.ceil(max(pt)/args.max_step));ph=pt/pn;py=np.array([q['peak'] for q in probe]);pstart=py.copy();pmin=py.copy();pmax=py.copy()
   def pf(z):return np.clip(np.nan_to_num(design(np.clip(z,-3,4),pg,kn)@co,nan=100,posinf=100,neginf=-100),-50,50)
   for _ in range(pn):
    k1=pf(py);k2=pf(py+ph[:,None]*k1/2);k3=pf(py+ph[:,None]*k2/2);k4=pf(py+ph[:,None]*k3);py=py+ph[:,None]*(k1+2*k2+2*k3+k4)/6;pmin=np.minimum(pmin,py);pmax=np.maximum(pmax,py)
   full_pen=np.r_[4*np.maximum(-pmin[:,2],0)/.1,2*np.maximum(-pmin[:,0],0)/.1,(np.maximum(-pmin[:,1],0)+np.maximum(pmax[:,1]-1,0))/.1, 1.5*((py-pstart)/scale).ravel(),2*(pmax[:,0]-np.array([q['Cmax'] for q in probe]))/.4,2*(pmin[:,0]-np.array([q['Cmin'] for q in probe]))/.4]/np.sqrt(len(probe))
  mats=matrices(co);lam=np.array([max(np.linalg.eigvals(z).real) for z in mats]);other=np.maximum(.01-np.array([1.,-1.])*lam[2:],0)/.2
  z=np.r_[ferr.ravel()/np.sqrt(ferr.size),2*err.ravel()/np.sqrt(err.size),3*negI/np.sqrt(len(negI)),2*negC/np.sqrt(len(negC)),badH/np.sqrt(len(badH)),full_pen,.2*other,.001*(v-v0)]
  z=np.clip(np.nan_to_num(z,nan=1e4,posinf=1e4,neginf=-1e4),-1e4,1e4);cost=float(z@z)
  # Track only feasible evaluated points; the final result is checked again below.
  feasible=np.min(constraints(v))>=-1e-7
  if feasible and cost<best[0]:best[:]=[cost,v.copy()]
  if calls%75==1:print('calls',calls,'cost',round(cost,5),'feasible',feasible,'best',round(best[0],5),'elapsed',round(time.monotonic()-start,1),flush=True)
  return cost
 if np.min(constraints(v0)) < 0:raise RuntimeError('starting candidate must satisfy stability constraints in all four cells')
 perturbed=False
 if args.seed:
  rng=np.random.default_rng(args.seed)
  for _ in range(100):
   trial=v0+rng.normal(0,.002,size=v0.shape)
   if np.min(constraints(trial))>=0:v0=trial;perturbed=True;break
 loss(v0)
 sol=minimize(loss,v0,method='SLSQP',constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':args.maxiter,'ftol':1e-6,'disp':True,'finite_diff_rel_step':1e-5})
 if np.min(constraints(sol.x))>=-1e-7:loss(sol.x)
 if best[1] is None:raise RuntimeError('no feasible candidate found')
 co=N@best[1].reshape(shape);A=matrices(co);leads=[float(max(np.linalg.eigvals(z).real)) for z in A];out={'mode':'FM','knots':kn.tolist(),'coefficients':co.tolist(),'fixed_equilibria':d['fixed_equilibria'],'max_anchor_residual':float(np.max(np.abs(design(anchors,ag,kn)@co))),'training_gammas':gg.tolist(),'objective':best[0],'calls':calls,'seed':args.seed,'start_perturbed':perturbed,'status':str(sol.message),'optimizer_success':bool(sol.success),'lower_zone_leading_eigenvalues':leads[:2],'all_zone_leading_eigenvalues':leads,'max_step':float(max(dt)),'substeps_per_segment':ns,'full_period_probes':bool(args.full_probes),'full_probe_gammas':[float(q['gamma']) for q in probe] if args.full_probes else [],'stability_constraint':'max real eigenvalue <= -0.001 in both lower cells and upper cell, >= 0.001 in middle cell; numerically checked to tolerance 1e-7','limitations':'Finite-step, clipped multiple-shooting proxy; IP3 and physical-state penalties are soft; no proof of global positivity, stable cycles, HLB subtype or correct equilibrium branch. Check adaptive validation and equilibrium crossings.'};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n');print('DONE',best[0],leads,sol.message,flush=True)
if __name__=='__main__':main()
