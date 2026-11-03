#!/usr/bin/env python3
"""Explore reference G-ChI equilibria and attracting cycles, independently for each mode."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import root, brentq
from scipy.integrate import solve_ivp
P={'AFM':dict(ker=.1,vdelta=.02,r5=.04,vbeta=.2),'FM':dict(ker=.05,vdelta=.05,r5=.05,vbeta=.5)}
def field(t,y,g,p):
 c,h,i=y; q=1.049*(i+.13)/(i+.9434); m=i/(i+.13); n=c/(c+.08234)
 return np.array([(6*m**3*n**3*h**3+.11)*(2-1.185*c)-.9*c*c/(c*c+p['ker']**2),.2*(q-(q+c)*h),p['vbeta']*g**.7/(g**.7+(1.3+10*c/(c+.6))**.7)+p['vdelta']/(1+i/1.5)*c*c/(c*c+.1**2)-2*c**4/(c**4+.7**4)*i/(i+1)-p['r5']*i])
def jac(y,g,p):
 return np.column_stack([np.imag(field(0,np.asarray(y,dtype=complex)+1e-24j*np.eye(3)[i],g,p))/1e-24 for i in range(3)])
def roots(g,p,starts):
 found=[]
 for z in starts:
  z=root(lambda y:field(0,y,g,p),z,jac=lambda y:jac(y,g,p),tol=1e-10)
  if np.linalg.norm(z.fun)>1e-8 or not np.all(np.isfinite(z.x)) or np.min(z.x)<0 or z.x[0]>2 or z.x[1]>1.01 or z.x[2]>4:continue
  if all(np.linalg.norm(z.x-q)>1e-5 for q in found):found.append(z.x)
 return sorted(found,key=lambda z:z[0])
def simulate(g,p,y0,duration):
 def f(t,y):return field(t,y,g,p)
 def peak(t,y):return f(t,y)[0]
 peak.direction=-1
 s=solve_ivp(f,(0,duration),y0,method='DOP853',events=peak,dense_output=True,rtol=1e-8,atol=1e-10,max_step=.5)
 if not s.success:return {'status':'failed','end':s.y[:,-1].tolist()}
 peaks=s.t_events[0];peaks=peaks[peaks>duration*.45]
 if len(peaks)<4:return {'status':'undetermined','end':s.y[:,-1].tolist()}
 ts=np.diff(peaks[-4:]); a=s.sol(peaks[-1]);b=s.sol(peaks[-2]); y=s.sol(np.linspace(peaks[-2],peaks[-1],201)); err=np.linalg.norm(a-b); amp=np.ptp(y[0]);status='cycle' if amp>1e-4 and np.max(abs(ts-ts[-1]))/ts[-1]<.003 and err<max(1e-5,.003*amp) else 'undetermined'
 return {'status':status,'end':s.y[:,-1].tolist(),'peak':a.tolist(),'period':float(ts[-1]),'Cmin':float(y[0].min()),'Cmax':float(y[0].max()),'Imin':float(y[2].min()),'Imax':float(y[2].max()),'closure':float(err)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=P,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--points',type=int,default=40);ap.add_argument('--duration',type=float,default=800);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);p=P[a.mode]; E=[]; cycles=[];prev=[];y=[.2,.7,.4]
 for g in np.r_[0,np.geomspace(.001,20,a.points)]:
  starts=prev+[np.array([c,h,i]) for c in (.03,.08,.16,.3,.55,.8) for h in (.4,.8) for i in (.15,.5,1.0)]
  rr=roots(float(g),p,starts);prev=rr; rows=[]
  for x in rr:
   ev=np.linalg.eigvals(jac(x,float(g),p));rows.append({'state':x.tolist(),'eigenvalues':[[z.real,z.imag] for z in ev],'max_real':float(max(ev.real))})
  E.append({'gamma':float(g),'roots':rows})
  if g>0:
   cy=simulate(float(g),p,y,a.duration);cy['gamma']=float(g);cycles.append(cy);y=cy['end'];print(a.mode,round(float(g),5),'roots',len(rr),cy['status'],flush=True)
 (a.out/'reference.json').write_text(json.dumps({'mode':a.mode,'parameters':p,'equilibria':E,'cycles':cycles,'limitations':'Equilibria found by multistart, stable cycles by forward simulations only; neither is exhaustive.'},indent=2)+'\n')
if __name__=='__main__':main()
