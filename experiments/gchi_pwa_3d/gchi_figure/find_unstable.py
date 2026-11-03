import sys,json
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import root
R=Path(__file__).resolve().parent.parent / 'gchi_modes';sys.path.insert(0,str(R));from explore_modes import field,jac,P
f0=json.load(open(R/'analysis/fm-constrained-01/reference-fold-candidate.json')); base=np.r_[f0['state'],f0['period']];p=P['FM']
def solve(g,guess):
 def res(x):
  if x[3]<4 or x[3]>80 or min(x[:3])<0 or max(x[:3])>10:return np.ones(4)*1e3
  sol=solve_ivp(field,(0,x[3]),x[:3],args=(g,p),rtol=5e-10,atol=1e-12,method='DOP853',max_step=.12)
  if not sol.success:return np.ones(4)*1e3
  return np.r_[sol.y[:,-1]-x[:3],field(0,x[:3],g,p)[0]]
 z=root(res,guess,tol=1e-8)
 if np.linalg.norm(res(z.x))>1e-7:return None
 x=z.x
 def rhs(t,u):return np.r_[field(t,u[:3],g,p),(jac(u[:3],g,p)@u[3:].reshape(3,3)).ravel()]
 a=solve_ivp(rhs,(0,x[3]),np.r_[x[:3],np.eye(3).ravel()],method='DOP853',rtol=1e-10,atol=1e-12,max_step=.08,dense_output=True);ev=np.linalg.eigvals(a.y[3:,-1].reshape(3,3));neutral=np.argmin(abs(ev-1));ev=np.delete(ev,neutral);rho=max(abs(ev));y=a.sol(np.linspace(0,x[3],800))[:3];return {'gamma':g,'state':x[:3].tolist(),'period':x[3],'Cmin':float(min(y[0])),'Cmax':float(max(y[0])),'Imin':float(min(y[2])),'Imax':float(max(y[2])),'rho':float(rho),'closure':float(np.linalg.norm(a.y[:3,-1]-x[:3]))}
results=[];g=f0['gamma']-.001
x=base[:3]
def variational(t,u):return np.r_[field(t,u[:3],f0['gamma'],p),(jac(u[:3],f0['gamma'],p)@u[3:].reshape(3,3)).ravel()]
s=solve_ivp(variational,(0,base[3]),np.r_[x,np.eye(3).ravel()],rtol=1e-10,atol=1e-12,method='DOP853',max_step=.08)
M=s.y[3:,-1].reshape(3,3);df=jac(x,f0['gamma'],p)[0];F=field(0,x,f0['gamma'],p);proj=np.eye(3)-np.outer(F,df)/(df@F);e,v=np.linalg.eig(proj@M);direction=np.real(v[:,np.argmin(abs(e-1))]);direction/=np.linalg.norm(direction)
for dh in [-.02,-.01,-.005,.005,.01,.02]:
 z=base.copy();z[:3]+=dh*direction;q=solve(g,z)
 if q:print('try',dh,'rho',q['rho'],'amp',q['Cmax']-q['Cmin'],flush=True)
 if q and q['rho']>1.00001 and q['Cmax']-q['Cmin']>.01:results.append(q);break
if results:
 for g in f0["gamma"]-np.geomspace(.0015,8,35):
  prev=results[-1];q=solve(float(g),np.r_[prev['state'],prev['period']]);
  if not q or q['rho']<=1.00001 or q['Cmax']-q['Cmin']<.01:print('stop',g,flush=True);break
  results.append(q);print('point',g,q['rho'],flush=True)
Path(str(Path(__file__).resolve().parent / 'fm_reference_unstable.json')).write_text(json.dumps(results,indent=2)+'\n');print('count',len(results),flush=True)
