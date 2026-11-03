#!/usr/bin/env python3
"""Pseudo-arclength continuation of periodic PWA solutions with Floquet analysis."""
import os,json,argparse
os.environ.setdefault('PWA_FIT_FILE','fit_actual_rest.json')
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from barrier_horizontal import f as guarded, load
ap=argparse.ArgumentParser(description='PWA periodic-orbit continuation');ap.add_argument('candidate',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--start-low',type=float,default=2.5);ap.add_argument('--start-high',type=float,default=3.0);args=ap.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
co,geom=load(str(args.candidate.resolve()))
fit=json.loads(args.candidate.read_text());cb=fit['barrier']['Cb'];k=fit['barrier']['k']
def fun(t,y,g): return guarded(t,y,g,co,geom,cb,k)
P=Path(__file__).parent

def mat(y):
 C,h,I=y;c0,i1,width,height=geom;top=c0+height;a=0
 A=np.column_stack((co[1],co[2],co[3])).copy()
 if C<cb:A[0,0]-=k
 if C>c0:A[:,0]+=co[7]
 if C>top:A[:,0]+=co[8];A[:,2]-=co[8]*a
 return A

def period_res(x,g):
 c,h,i,T=x
 if T<2 or T>30 or min(c,h,i)<0:return np.ones(4)*10
 y=np.array([c,h,i]);s=solve_ivp(fun,(0,T),y,args=(g,),rtol=1e-10,atol=1e-12,max_step=.05)
 return np.r_[s.y[:,-1]-y,fun(0,y,g)[0]]

def seed(g,init=None):
 def peak(t,y,g):return fun(t,y,g)[0]
 peak.direction=-1;peak.terminal=False
 s=solve_ivp(fun,(0,800),[.55,.63,.62] if init is None else init,args=(g,),events=peak,rtol=1e-8,atol=1e-10,max_step=.2,dense_output=True)
 if len(s.t_events[0])<4:raise RuntimeError('No peaks')
 amp=np.ptp(s.sol(np.linspace(s.t_events[0][-2],s.t_events[0][-1],120))[0]);
 if amp<.005:raise RuntimeError(f'Spurious equilibrium seed at gamma={g}: amplitude={amp}')
 x=np.r_[s.y_events[0][-1],np.diff(s.t_events[0][-2:])[-1]]
 z=root(lambda x:period_res(x,g),x,tol=1e-8)
 if np.linalg.norm(z.fun)>1e-6:raise RuntimeError(('seed',g,z.message,z.fun))
 return z.x

scale=np.array([.3,.3,.5,8,3]) # C,h,I,T,gamma
x1=seed(args.start_low);x2=seed(args.start_high);v0=np.r_[x1,args.start_low]/scale;v1=np.r_[x2,args.start_high]/scale
out=[]
for step_index in range(140):
 tangent=v1-v0;tangent/=np.linalg.norm(tangent);ds=.014
 for retry in range(8):
  pred=v1+ds*tangent
  def objective(v):
   c,h,i,T,g=v*scale
   if g<=0:return np.ones(5)*10
   return np.r_[period_res([c,h,i,T],g),np.dot(v-pred,tangent)]
  r=root(objective,pred,tol=1e-7,options={'eps':1e-6})
  if np.linalg.norm(r.fun)<=1e-6 and np.linalg.norm(r.x-pred)<=2*ds:break
  ds*=.5
 if np.linalg.norm(r.fun)>1e-6 or np.linalg.norm(r.x-pred)>2*ds:print('STOP',step_index,r.message,'res',np.linalg.norm(r.fun),'predictor_error',np.linalg.norm(r.x-pred),'ds',ds,flush=True);break
 v0,v1=v1,r.x;c,h,i,T,g=v1*scale;y=np.array([c,h,i])
 def rhs(t,u):return np.r_[fun(t,u[:3],g),(mat(u[:3])@u[3:].reshape(3,3)).ravel()]
 s=solve_ivp(rhs,(0,T),np.r_[y,np.eye(3).ravel()],method='DOP853',rtol=1e-9,atol=1e-11,max_step=.07,dense_output=True)
 ev=np.linalg.eigvals(s.y[3:,-1].reshape(3,3));ev=sorted(ev,key=lambda v:abs(v-1));rho=max(abs(a) for a in ev[1:]);a=s.sol(np.linspace(0,T,450))[:3]
 out.append({'gamma':float(g),'state':y.tolist(),'period':float(T),'Cmin':float(a[0].min()),'Cmax':float(a[0].max()),'Imin':float(a[2].min()),'Imax':float(a[2].max()),'rho':float(rho),'multipliers':[[float(z.real),float(z.imag)] for z in ev],'closure':float(np.linalg.norm(s.y[:3,-1]-y)),'residual':float(np.linalg.norm(r.fun))})
 args.out.write_text(json.dumps(out,indent=2)+'\n')
 if step_index%10==0:print('step',step_index,'gamma',round(g,6),'Cmax',round(a[0].max(),6),'rho',round(rho,6),'closure',round(out[-1]['closure'],9),flush=True)
 if g<2.08:break
 if out[-1]['Cmax']-out[-1]['Cmin']<.005:print('STOP: near equilibrium',flush=True);break
args.out.write_text(json.dumps(out,indent=2)+'\n')
