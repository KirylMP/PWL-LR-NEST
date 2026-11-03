import json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
R=Path(__file__).resolve().parent.parent / 'gchi_modes';sys.path.insert(0,str(R));from fit_modes import design
p=json.load(open(R/'runs/afm-calcium-c1-01/candidate-2.json'));co=np.array(p['coefficients']);kn=np.array(p['knots']);gamma=3.5
def F(t,y):return design(y,gamma,kn)[0]@co
def residual(v):
 if v[3]<2 or v[3]>30 or min(v[:3])<0 or max(v[:3])>5:return np.full(4,50.)
 s=solve_ivp(F,(0,v[3]),v[:3],rtol=2e-8,atol=2e-10,method='DOP853',max_step=.2)
 return np.r_[s.y[:,-1]-v[:3],F(0,v[:3])[0]]
for h in [.59,.61,.63,.65]:
 for i in [.65,.72,.8]:
  v=np.array([.55,h,i,8.]);sol=root(residual,v,tol=1e-7,options={'maxfev':80});err=np.linalg.norm(residual(sol.x));
  if err>1e-6:continue
  x=sol.x;y=solve_ivp(F,(0,x[3]),x[:3],rtol=1e-9,atol=1e-11,method='DOP853',max_step=.08,dense_output=True);amp=np.ptp(y.y[0]);
  if amp<.01:continue
  print('candidate',x,'closure',err,'amp',amp,flush=True)
  from scipy.optimize._numdiff import approx_derivative
  J=approx_derivative(lambda q:solve_ivp(F,(0,x[3]),q,rtol=1e-9,atol=1e-11,method='DOP853',max_step=.08).y[:,-1],x[:3],method='3-point');ev=np.linalg.eigvals(J);ev=np.delete(ev,np.argmin(abs(ev-1)));rho=float(max(abs(ev)));print('rho',rho,flush=True)
  if rho>1.001:
   vals=y.sol(np.linspace(0,x[3],1000));res={'gamma':gamma,'state':x[:3].tolist(),'period':float(x[3]),'Cmin':float(min(vals[0])),'Cmax':float(max(vals[0])),'Imin':float(min(vals[2])),'Imax':float(max(vals[2])),'rho':rho,'closure':err};results=[res];base=x.copy()
   for direction in (-1,1):
    x=base.copy()
    for target in np.arange(3.5+direction*.04, 2.0 if direction<0 else 4.3, direction*.04):
     gamma=float(target); sol=root(residual,x,tol=1e-7,options={'maxfev':90}); err=np.linalg.norm(residual(sol.x))
     if err>1e-6:break
     x=sol.x;y=solve_ivp(F,(0,x[3]),x[:3],rtol=1e-9,atol=1e-11,method='DOP853',max_step=.08,dense_output=True); vals=y.sol(np.linspace(0,x[3],700));
     if np.ptp(vals[0])<.001:break
     J=approx_derivative(lambda q:solve_ivp(F,(0,x[3]),q,rtol=1e-9,atol=1e-11,method='DOP853',max_step=.08).y[:,-1],x[:3],method='3-point');ev=np.linalg.eigvals(J);ev=np.delete(ev,np.argmin(abs(ev-1)));rho=float(max(abs(ev)))
     if rho<=1.001:break
     results.append({'gamma':gamma,'state':x[:3].tolist(),'period':float(x[3]),'Cmin':float(min(vals[0])),'Cmax':float(max(vals[0])),'Imin':float(min(vals[2])),'Imax':float(max(vals[2])),'rho':rho,'closure':err});print('point',gamma,rho,flush=True)
   open(str(Path(__file__).resolve().parent / 'am_pwa_unstable.json'),'w').write(json.dumps(results,indent=2));sys.exit(0)
print('no unstable root found')
