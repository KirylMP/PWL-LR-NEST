#!/usr/bin/env python3
"""Exploratory continuous PWA fit with rest and transition points pinned exactly."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import least_squares, root, brentq
from explore_modes import P,field,jac,roots

def design(y,g,knots):
 y=np.atleast_2d(y);g=np.broadcast_to(g,len(y));v=np.maximum(g,0)**.7
 return np.column_stack([np.ones(len(y)),y,v/(v+.02**.7),v/(v+.5**.7),v/(v+3**.7),*[np.maximum(y[:,0]-q,0) for q in knots]])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=P,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--nfev',type=int,default=80);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--jitter',type=float,default=.015);ap.add_argument('--start',type=Path);a=ap.parse_args();ref=json.loads(a.data.read_text());p=P[a.mode]
 # Three anchors: zero-glutamate rest and two reference transition points.
 if a.mode=='FM':
  def saddle_eq(z):return np.r_[field(0,z[:3],z[3],p),np.linalg.det(jac(z[:3],z[3],p))]
  s=root(saddle_eq,[.05385,.8975,.5348,.02015],tol=1e-11)
  if not s.success or np.linalg.norm(s.fun)>1e-8:raise RuntimeError('reference saddle-node not resolved')
  def eq(g):
   found=roots(g,p,[[.42,.59,.95]])
   if not found:raise RuntimeError('upper reference equilibrium not found')
   return found[-1]
  gh=brentq(lambda g:max(np.linalg.eigvals(jac(eq(g),g,p)).real),1.,10.,xtol=1e-11)
  ag=np.array([0.,s.x[3],gh]); anchors=np.vstack([ref['equilibria'][0]['roots'][0]['state'],s.x[:3],eq(gh)])
  knots=np.array([(anchors[0,0]+anchors[1,0])/2,anchors[1,0],anchors[2,0]])
 else:
  # AFM reference transition points from the earlier checked AM-derived G-ChI run.
  hopfs=json.loads((Path(__file__).parent.parent/'gchi_am/output/hopf.json').read_text())
  ag=np.array([0.,*[h['gamma'] for h in hopfs]])
  anchors=np.array([ref['equilibria'][0]['roots'][0]['state'],*[[h[k] for k in ('C','h','I')] for h in hopfs]])
  knots=np.array([(anchors[0,0]+anchors[1,0])/2,anchors[1,0],anchors[2,0]])
 D=design(anchors,ag,knots);N=null_space(D)
 good=[x for x in ref['cycles'] if x['status']=='cycle'];idx=np.unique(np.linspace(1,len(good)-2,min(9,len(good)-2),dtype=int));train=[good[i] for i in idx]
 states=[];gammas=[];derivs=[]; eqs=[]
 for r in train:
  g=r['gamma'];T=r['period'];s=solve_ivp(field,(0,T),r['peak'],args=(g,p),t_eval=np.linspace(0,T,40,endpoint=False),method='DOP853',rtol=1e-9,atol=1e-11,max_step=.2)
  if not s.success:raise RuntimeError('reference integration failed')
  states.extend(s.y.T);gammas.extend([g]*len(s.t));derivs.extend([field(0,x,g,p) for x in s.y.T])
 for r in ref['equilibria']:
  
  if r['roots']:
   q=r['roots'][0] if r['gamma'] < ag[1] else r['roots'][-1]
   eqs.append((r['gamma'],q['state']))
 X=np.array(states);G=np.array(gammas);Y=np.array(derivs);XE=np.array([q[1] for q in eqs]);GE=np.array([q[0] for q in eqs]);B=design(X,G,knots)@N;BE=design(XE,GE,knots)@N
 A=np.vstack([B/np.sqrt(len(B)),2*BE/np.sqrt(len(BE))]);b=np.vstack([Y/np.sqrt(len(B)),np.zeros((len(BE),3))]);V=np.linalg.lstsq(A,b,rcond=None)[0]
 yinit=np.array([q['peak'] for q in train]);gg=np.array([q['gamma'] for q in train]);period=np.array([q['period'] for q in train]); phases=[]
 for q in train:
  s=solve_ivp(field,(0,q['period']),q['peak'],args=(q['gamma'],p),t_eval=np.linspace(0,q['period'],9),rtol=1e-9,atol=1e-11,method='DOP853',max_step=.2);phases.append(s.y.T)
 phases=np.array(phases);scale=np.array([.4,.15,.4]);counter=0;best=[float('inf'),None]
 def loss(v):
  nonlocal counter,best
  counter+=1;co=N@v.reshape(V.shape);y=phases[:,:-1,:].reshape(-1,3).copy();seg_g=np.repeat(gg,8);h=np.repeat(period/160,8)
  def f(z):return np.clip(np.nan_to_num(design(np.clip(z,-2,3),seg_g,knots)@co,nan=100,posinf=100,neginf=-100),-20,20)
  for j in range(20):
   k1=f(y);k2=f(y+h[:,None]*k1/2);k3=f(y+h[:,None]*k2/2);k4=f(y+h[:,None]*k3);y=y+h[:,None]*(k1+2*k2+2*k3+k4)/6
  err=(y.reshape(len(gg),8,3)-phases[:,1:,:])/scale;fc=((B@v.reshape(V.shape)-Y)/np.array([.4,.05,.1])).ravel()/np.sqrt(len(B)); ec=((BE@v.reshape(V.shape))/np.array([.4,.05,.1])).ravel()/np.sqrt(len(BE));leads=[]
  for ci in [knots[0]/2,(knots[0]+knots[1])/2,(knots[1]+knots[2])/2,knots[2]+.05]:
   matrix=co[1:4,:].T.copy()
   for ki,knot in enumerate(knots):
    if ci>knot:matrix[:,0]+=co[7+ki]
   leads.append(float(max(np.linalg.eigvals(matrix).real)))
  spectral=np.maximum(.01-np.array([-1.,-1.,1.,-1.])*np.array(leads),0)/.2
  z=np.r_[fc,ec,1.5*err.ravel()/np.sqrt(err.size),.75*spectral,.002*(v-V.ravel())];z=np.clip(np.nan_to_num(z,nan=1e4,posinf=1e4,neginf=-1e4),-1e4,1e4);val=float(z@z)
  if val<best[0]:best=[val,v.copy()]
  if counter%100==1:print(a.mode,'calls',counter,'loss',round(val,5),'best',round(best[0],5),flush=True)
  return z
 vstart=V.ravel().copy()
 if a.start:
  prior=json.loads(a.start.read_text())
  if prior['mode']!=a.mode or not np.allclose(prior['knots'],knots,rtol=0,atol=1e-8):raise ValueError('start mode or knots differ')
  vstart=(N.T@np.array(prior['coefficients'])).ravel()
 if a.seed:vstart+=np.random.default_rng(a.seed).normal(0,a.jitter,size=vstart.shape)
 sol=least_squares(loss,vstart,max_nfev=a.nfev,ftol=1e-5,xtol=1e-5,x_scale='jac');co=N@best[1].reshape(V.shape)
 out={'mode':a.mode,'knots':knots.tolist(),'coefficients':co.tolist(),'fixed_equilibria':[{'gamma':float(g),'state':x.tolist()} for g,x in zip(ag,anchors)],'max_anchor_residual':float(np.max(np.abs(D@co))),'training_gammas':gg.tolist(),'objective':best[0],'calls':counter,'seed':a.seed,'jitter':a.jitter,'starting_candidate':str(a.start) if a.start else None,'status':sol.message,'input_exponent':.7,'stability_target':'soft penalty: stable below first transition, unstable between transitions, stable above second', 'limitations':'Reference rest and transition equilibrium positions are fixed, not the smooth bifurcation types. Other branches are not enforced. No hard constraint on stability, positivity or fold; validate independently.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print('DONE',a.mode,best[0],flush=True)
if __name__=='__main__':main()
