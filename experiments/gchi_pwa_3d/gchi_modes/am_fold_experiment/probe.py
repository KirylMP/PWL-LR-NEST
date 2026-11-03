import sys,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
R=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R));from fit_modes import design
D=json.load(open(R/'runs/afm-calcium-c1-01/candidate-2.json'));k=np.array(D['knots']);co0=np.array(D['coefficients']);anc=np.array([x['state'] for x in D['fixed_equilibria']]);ag=np.array([x['gamma'] for x in D['fixed_equilibria']]);N=null_space(design(anc,ag,k));w0=N.T@co0;gamma=3.7500176522854862
branches=json.load(open(str(R.parent / 'gchi_figure/am_pwa_unstable.json')));q=min(branches,key=lambda x:abs(x['gamma']-gamma));initial=np.r_[q['state'],q['period']]; ref=json.load(open(R.parent/'gchi_am/output/fold.json')); refvec=np.array([ref['Cmin'],ref['Cmax'],ref['Imin'],ref['Imax'],ref['period']]);print('start',initial,'rho',q['rho_variational'],flush=True)
points=np.linspace(0,1,180); evals=[0];t0=time.monotonic()
def obj(v):
 evals[0]+=1;co=N@v[:21].reshape(7,3);y0=v[21:24];T=v[24]
 def rhs(t,u):
  y=u[:3];zone=np.searchsorted(k,y[0],side='right');A=co[1:4].T.copy();A[:,0]+=np.sum(co[7:7+zone],axis=0) if zone else 0
  return np.r_[design(y,gamma,k)[0]@co,(A@u[3:].reshape(3,3)).ravel()]
 try:
  s=solve_ivp(rhs,(0,T),np.r_[y0,np.eye(3).ravel()],rtol=3e-7,atol=3e-9,method='DOP853',max_step=.1,dense_output=True)
  if not s.success or np.max(abs(s.y))>1e6: raise ValueError
  y=s.sol(points*T)[:3];M=s.y[3:,-1].reshape(3,3);tr=np.trace(M);pair=(tr*tr-np.trace(M@M))/2;fold=3-2*tr+pair;ext=np.array([min(y[0]),max(y[0]),min(y[2]),max(y[2]),T]);reg=(v[:21]-w0.ravel())*.005
  res=np.r_[20*(s.y[:3,-1]-y0),10*(design(y0,gamma,k)[0]@co)[0],2*fold,2*(ext-refvec)/np.array([.2,.2,.2,.2,5]),reg]
  if not np.all(np.isfinite(res)):raise ValueError
 except (ValueError,OverflowError,FloatingPointError):res=np.ones(31)*100
 if evals[0]%80==1:print('calls',evals[0],'cost',np.linalg.norm(res),'t',round(time.monotonic()-t0,1),flush=True)
 return res
v0=np.r_[w0.ravel(),initial];lo=np.r_[w0.ravel()-5,[.1,.05,.05,3.]];hi=np.r_[w0.ravel()+5,[1.4,.99,2,20.]]
sol=least_squares(obj,v0,bounds=(lo,hi),max_nfev=30,ftol=1e-6,xtol=1e-6,diff_step=1e-5,verbose=1);co=N@sol.x[:21].reshape(7,3);out={'source':'runs/afm-calcium-c1-01/candidate-2.json','gamma':gamma,'coefficients':co.tolist(),'knots':k.tolist(),'fixed_equilibria':D['fixed_equilibria'],'state':sol.x[21:24].tolist(),'period':float(sol.x[24]),'cost':float(np.linalg.norm(sol.fun)),'optimizer_success':bool(sol.success),'status':str(sol.message),'nfev':sol.nfev,'anchor_residual':float(np.max(abs(design(anc,ag,k)@co))),'note':'soft shooting and fold proxy only; requires independent validation'}
Path(str(Path(__file__).resolve().parent / 'pilot.json')).write_text(json.dumps(out,indent=2));print('DONE',json.dumps({k:out[k] for k in ['cost','status','anchor_residual','state','period']}),flush=True)
