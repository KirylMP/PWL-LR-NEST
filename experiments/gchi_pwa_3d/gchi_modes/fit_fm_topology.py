#!/usr/bin/env python3
"""FM fit with pinned anchors, stable regions, onset probes and a fold shooting proxy."""
import argparse,json,time,os,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import minimize
from fit_modes import design
from explore_modes import P,field

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--start',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--maxiter',type=int,default=80);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--max-step',type=float,default=.035);ap.add_argument('--full-probes',action='store_true');ap.add_argument('--onset-data',type=Path,required=True);ap.add_argument('--fold-target',type=Path,required=True);ap.add_argument('--resume',type=Path,help='resume from a compatible per-seed checkpoint');args=ap.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
 d=json.loads(args.start.read_text());ref=json.loads(args.data.read_text());kn=np.array(d['knots']);anchors=np.array([x['state'] for x in d['fixed_equilibria']]);ag=np.array([x['gamma'] for x in d['fixed_equilibria']]);N=null_space(design(anchors,ag,kn));base=np.array(d['coefficients']);v0=(N.T@base).ravel();shape=(N.shape[1],3);p=P['FM'];nc=v0.size;fold=json.loads(args.fold_target.read_text());fy0=np.array(fold['state']);ft0=float(fold['period']);fg=float(fold['gamma']);v0=np.r_[v0,fy0,ft0];fold_lb=np.array([.7,.4,.5,8.]);fold_ub=np.array([1.5,.95,4.,25.]);vbase=v0.copy()
 good=[x for x in ref['cycles'] if x['status']=='cycle'];idx=np.unique(np.linspace(1,len(good)-2,min(9,len(good)-2),dtype=int));train=[good[i] for i in idx];near=json.loads(args.onset_data.read_text());near=[dict(z['result'],gamma=z['gamma']) for z in near['reference_low'] if z['result']['status']=='cycle'];train+=near;phases=[];X=[];G=[];Y=[]
 for q in train:
  g=q['gamma'];T=q['period'];s=solve_ivp(field,(0,T),q['peak'],args=(g,p),t_eval=np.linspace(0,T,41),method='DOP853',rtol=1e-9,atol=1e-11,max_step=.15)
  if not s.success:raise RuntimeError('reference integration failed')
  phases.append(s.y[:,::5].T);X.extend(s.y.T[:-1]);G.extend([g]*40);Y.extend([field(0,z,g,p) for z in s.y.T[:-1]])
 probe=[train[j] for j in (0,1,3,6,8)]
 fingerprint={'reference':ref,'start':d,'onset':near,'fold':fold,'max_step':args.max_step,'full_probes':args.full_probes};code=b''.join(Path(__file__).with_name(x).read_bytes() for x in ('fit_fm_topology.py','fit_modes.py','explore_modes.py'));signature=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()+code).hexdigest()
 phases=np.array(phases);gg=np.array([q['gamma'] for q in train]);period=np.array([q['period'] for q in train]);ns=max(20,int(np.ceil(max(period)/(8*args.max_step))));dt=np.repeat(period/(8*ns),8);gs=np.repeat(gg,8);X=np.array(X);G=np.array(G);Y=np.array(Y);B=design(X,G,kn)@N;scale=np.array([.4,.15,.4]);calls=0;best=[np.inf,None];start=time.monotonic()
 def matrices(co):
  A=co[1:4].T.copy();out=[A.copy()]
  for i in range(3):A=A.copy();A[:,0]+=co[7+i];out.append(A)
  return out
 def constraints(v):
  co=N@v[:nc].reshape(shape);A=matrices(co);leads=np.array([max(np.linalg.eigvals(m).real) for m in A])
  return np.array([-.001-leads[0],-.001-leads[1],leads[2]-.001,-.001-leads[3]])
 def loss(v):
  nonlocal calls,best
  calls+=1;co=N@v[:nc].reshape(shape);y=phases[:,:-1,:].reshape(-1,3).copy();mins=y.copy();maxh=y[:,1].copy()
  def f(z):return np.clip(np.nan_to_num(design(np.clip(z,-3,4),gs,kn)@co,nan=100,posinf=100,neginf=-100),-50,50)
  for _ in range(ns):
   k1=f(y);k2=f(y+dt[:,None]*k1/2);k3=f(y+dt[:,None]*k2/2);k4=f(y+dt[:,None]*k3);y=y+dt[:,None]*(k1+2*k2+2*k3+k4)/6;mins=np.minimum(mins,y);maxh=np.maximum(maxh,y[:,1])
  err=(y.reshape(len(gg),8,3)-phases[:,1:,:])/scale;pred=B@v[:nc].reshape(shape);ferr=(pred-Y)/np.array([.4,.05,.1]);negI=np.maximum(-mins[:,2],0)/.1;negC=np.maximum(-mins[:,0],0)/.1;badH=(np.maximum(-mins[:,1],0)+np.maximum(maxh-1,0))/.1
  full_pen=np.zeros(50)
  if args.full_probes:
   pg=np.array([q['gamma'] for q in probe]);pt=np.array([q['period'] for q in probe]);pn=int(np.ceil(max(pt)/args.max_step));ph=pt/pn;py=np.array([q['peak'] for q in probe]);pstart=py.copy();pmin=py.copy();pmax=py.copy()
   def pf(z):return np.clip(np.nan_to_num(design(np.clip(z,-3,4),pg,kn)@co,nan=100,posinf=100,neginf=-100),-50,50)
   for _ in range(pn):
    k1=pf(py);k2=pf(py+ph[:,None]*k1/2);k3=pf(py+ph[:,None]*k2/2);k4=pf(py+ph[:,None]*k3);py=py+ph[:,None]*(k1+2*k2+2*k3+k4)/6;pmin=np.minimum(pmin,py);pmax=np.maximum(pmax,py)
   full_pen=np.r_[4*np.maximum(-pmin[:,2],0)/.1,2*np.maximum(-pmin[:,0],0)/.1,(np.maximum(-pmin[:,1],0)+np.maximum(pmax[:,1]-1,0))/.1, 1.5*((py-pstart)/scale).ravel(),2*(pmax[:,0]-np.array([q['Cmax'] for q in probe]))/.4,2*(pmin[:,0]-np.array([q['Cmin'] for q in probe]))/.4,2*(pmin[:,2]-np.array([q['Imin'] for q in probe]))/.5,2*(pmax[:,2]-np.array([q['Imax'] for q in probe]))/.5]/np.sqrt(len(probe))
  # Long onset orbit: soft closure and IP3/C extrema matching from the reference cycle peak.
  q=next(q for q in near if abs(q['gamma']-.021)<1e-8);steps=int(np.ceil(q['period']/args.max_step));delta=q['period']/steps;yn=np.array(q['peak']);yn0=yn.copy();nmin=yn.copy();nmax=yn.copy()
  def nf(z):return np.clip(np.nan_to_num(design(np.clip(z,-3,4),q['gamma'],kn)[0]@co,nan=100,posinf=100,neginf=-100),-50,50)
  for _ in range(steps):
   k1=nf(yn);k2=nf(yn+delta*k1/2);k3=nf(yn+delta*k2/2);k4=nf(yn+delta*k3);yn=yn+delta*(k1+2*k2+2*k3+k4)/6;nmin=np.minimum(nmin,yn);nmax=np.maximum(nmax,yn)
  onset_pen=np.r_[2*(yn-yn0)/scale,2*(nmin[[0,2]]-np.array([q['Cmin'],q['Imin']]))/np.array([.2,.3]),2*(nmax[[0,2]]-np.array([q['Cmax'],q['Imax']]))/np.array([.4,.3]),4*max(-nmin[2],0)/.1,2*max(-nmin[0],0)/.1,(max(-nmin[1],0)+max(nmax[1]-1,0))/.1]
  # Fold proxy: shooting closure, maximum phase, and a second unit Floquet multiplier.
  fy=v[nc:nc+3];ft=float(v[nc+3]);eps=1e-4;fz=np.vstack([fy,fy+eps*np.eye(3)]);fmin=fy.copy();fmax=fy.copy();fn=int(np.ceil(ft/args.max_step));fd=ft/fn
  def ff(z):return np.clip(np.nan_to_num(design(np.clip(z,-3,4),fg,kn)@co,nan=100,posinf=100,neginf=-100),-50,50)
  for _ in range(fn):
   k1=ff(fz);k2=ff(fz+fd*k1/2);k3=ff(fz+fd*k2/2);k4=ff(fz+fd*k3);fz=fz+fd*(k1+2*k2+2*k3+k4)/6;fmin=np.minimum(fmin,fz[0]);fmax=np.maximum(fmax,fz[0])
  mon=(fz[1:]-fz[0]).T/eps;tr=np.trace(mon);p2=(tr*tr-np.trace(mon@mon))/2;second=3-2*tr+p2
  fold_pen=np.r_[2*(fz[0]-fy)/scale,ff(fy)[0,0]/.5,np.clip(second,-10,10),2*(fmin[0]-fold['Cmin'])/.2,2*(fmax[0]-fold['Cmax'])/.4,2*(fmin[2]-fold['Imin'])/.6,2*(fmax[2]-fold['Imax'])/.6,.2*(ft-ft0)/ft0,4*max(-fmin[2],0)/.1,2*max(-fmin[0],0)/.1,(max(-fmin[1],0)+max(fmax[1]-1,0))/.1]
  mats=matrices(co);lam=np.array([max(np.linalg.eigvals(z).real) for z in mats]);other=np.maximum(.01-np.array([1.,-1.])*lam[2:],0)/.2
  z=np.r_[ferr.ravel()/np.sqrt(ferr.size),2*err.ravel()/np.sqrt(err.size),3*negI/np.sqrt(len(negI)),2*negC/np.sqrt(len(negC)),badH/np.sqrt(len(badH)),full_pen,onset_pen,fold_pen,.2*other,.001*(v-vbase)]
  z=np.clip(np.nan_to_num(z,nan=1e4,posinf=1e4,neginf=-1e4),-1e4,1e4);cost=float(z@z)
  # Track only feasible evaluated points; the final result is checked again below.
  feasible=np.min(constraints(v))>=-1e-7 and np.all(v[nc:]>=fold_lb-1e-10) and np.all(v[nc:]<=fold_ub+1e-10)
  if feasible and cost<best[0]:
   best[:]=[cost,v.copy()];tmp=Path(str(args.output)+'.tmp.npz');target=args.output.with_suffix('.checkpoint.npz')
   np.savez(tmp,v=v,cost=cost,calls=calls,knots=kn,anchors=anchors,anchor_gammas=ag,fold_gamma=fg,signature=signature);os.replace(tmp,target)
  if calls%75==1:print('calls',calls,'cost',round(cost,5),'feasible',feasible,'best',round(best[0],5),'elapsed',round(time.monotonic()-start,1),flush=True)
  return cost
 if args.resume:
  ck=np.load(args.resume,allow_pickle=False)
  if 'signature' not in ck or str(ck['signature'])!=signature or ck['v'].shape!=v0.shape or not np.allclose(ck['knots'],kn) or not np.allclose(ck['anchors'],anchors) or not np.allclose(ck['anchor_gammas'],ag) or not np.isclose(float(ck['fold_gamma']),fg):raise ValueError('incompatible checkpoint')
  v0=ck['v'].copy()
 if np.min(constraints(v0)) < -1e-7 or not np.all(v0[nc:]>=fold_lb) or not np.all(v0[nc:]<=fold_ub):raise RuntimeError('starting candidate must satisfy all stability and fold-state bounds')
 perturbed=False
 if args.seed and not args.resume:
  rng=np.random.default_rng(args.seed)
  for _ in range(100):
   trial=v0+rng.normal(0,.002,size=v0.shape)
   if np.min(constraints(trial))>=0 and np.all(trial[nc:]>=fold_lb) and np.all(trial[nc:]<=fold_ub):v0=trial;perturbed=True;break
 loss(v0)
 bounds=[(None,None)]*nc+list(zip(fold_lb,fold_ub))
 sol=minimize(loss,v0,method='SLSQP',bounds=bounds,constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':args.maxiter,'ftol':1e-6,'disp':True,'eps':1e-5})
 if np.min(constraints(sol.x))>=-1e-7:loss(sol.x)
 if best[1] is None:raise RuntimeError('no feasible candidate found')
 co=N@best[1][:nc].reshape(shape);A=matrices(co);leads=[float(max(np.linalg.eigvals(z).real)) for z in A];out={'mode':'FM','knots':kn.tolist(),'coefficients':co.tolist(),'fixed_equilibria':d['fixed_equilibria'],'max_anchor_residual':float(np.max(np.abs(design(anchors,ag,kn)@co))),'training_gammas':gg.tolist(),'fold_proxy':{'gamma':fg,'state':best[1][nc:nc+3].tolist(),'period':float(best[1][nc+3]),'method':'fixed-step one-period shooting; finite-difference monodromy; soft penalty only'},'onset_probe_gammas':[float(q['gamma']) for q in near],'objective':best[0],'calls':calls,'seed':args.seed,'fit_signature':signature,'resumed':bool(args.resume),'start_perturbed':perturbed,'status':str(sol.message),'optimizer_success':bool(sol.success),'lower_zone_leading_eigenvalues':leads[:2],'all_zone_leading_eigenvalues':leads,'max_step':float(max(dt)),'substeps_per_segment':ns,'full_period_probes':bool(args.full_probes),'full_probe_gammas':[float(q['gamma']) for q in probe] if args.full_probes else [],'stability_constraint':'max real eigenvalue <= -0.001 in both lower cells and upper cell, >= 0.001 in middle cell; numerically checked to tolerance 1e-7','limitations':'Finite-step clipped trajectory/fold proxy, all orbit and physical penalties soft; no guarantee of a fold or its nondegeneracy, long-period cycle, global positivity, HLB subtype or branch matching. Independent adaptive validation and numerical continuation are essential.'};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n');print('DONE',best[0],leads,sol.message,flush=True)
if __name__=='__main__':main()
