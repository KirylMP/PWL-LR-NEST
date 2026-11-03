import json,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.integrate import solve_ivp
from scipy.optimize import root
R=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R));from fit_modes import design
from poincare_fm_exact import evaluate_exact
from poincare_fm import NoReturn
p=json.load(open(str(Path(__file__).resolve().parent / 'pilot.json')));k=np.array(p['knots']);anc=np.array([x['state'] for x in p['fixed_equilibria']]);ag=np.array([x['gamma'] for x in p['fixed_equilibria']]);N=null_space(design(anc,ag,k));W=(N.T@np.array(p['coefficients'])).ravel();gamma=p['gamma'];co=N@W.reshape(7,3);sec=.20
f=lambda t,y:design(y,gamma,k)[0]@co
def section(t,y):return y[0]-sec
section.direction=1
s=solve_ivp(f,(0,20),p['state'],events=section,rtol=1e-10,atol=1e-12,method='DOP853',max_step=.01)
if len(s.y_events[0])==0:raise RuntimeError('No initial section')
u=s.y_events[0][0][1:];print('u',u,flush=True);calls=0

def evaluate(w,uv):
 global calls
 calls+=1
 try:
  r=evaluate_exact(N@w.reshape(7,3),k,gamma,uv,c_section=sec,tmax=30,scan_step=.06)
  residual=np.r_[r['return_state']-uv,r['fold_residual']]
  if not np.isfinite(residual).all():raise ValueError
  return residual,r
 except (ValueError,OverflowError,NoReturn):return np.ones(3)*100,None
base,ret=evaluate(W,u);print('base residual',base,'return',ret['return_time'] if ret else None,flush=True)
from scipy.optimize import least_squares
z0=np.r_[W,u]
def fun(z):
 res,rr=evaluate(z[:21],z[21:]);reg=.002*(z[:21]-W)
 if rr is None:return np.r_[np.full(3,100.),reg]
 return np.r_[res*np.array([10.,10.,2.]),reg]
sol=least_squares(fun,z0,max_nfev=80,diff_step=2e-5,xtol=1e-10,ftol=1e-10,gtol=1e-10,verbose=1)
w=sol.x[:21];uv=sol.x[21:];err,rr=evaluate(w,uv);print('least squares',err,'state',uv,'nfev',sol.nfev,flush=True)
if rr is not None:
 p.update(coefficients=(N@w.reshape(7,3)).tolist(),section_C=sec,section_state=uv.tolist(),return_time=rr['return_time'],fold_residual=err.tolist(),note='exact affine first return provisional; finite crossing scan and soft constraint, requires independent validation')
 Path(str(Path(__file__).resolve().parent / 'exact_candidate.json')).write_text(json.dumps(p,indent=2)+'\n')
