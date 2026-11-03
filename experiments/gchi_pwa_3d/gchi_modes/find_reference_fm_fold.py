import sys,json,argparse,numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import root
from explore_modes import field,jac,P
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
p=P['FM'];x0=np.r_[1.1512605668199103,.6807100478055937,2.284767193503642,12.416410091206174,21.202553866568334];calls=0
def residual(x):
 global calls
 calls+=1;y=x[:3];T=x[3];g=x[4]
 if T<3 or T>40 or not 10<g<40 or np.min(y)<0:return np.ones(5)*100
 def rhs(t,u):return np.r_[field(t,u[:3],g,p),(jac(u[:3],g,p)@u[3:].reshape(3,3)).ravel()]
 s=solve_ivp(rhs,(0,T),np.r_[y,np.eye(3).ravel()],method='DOP853',rtol=2e-11,atol=2e-13,max_step=.06)
 M=s.y[3:,-1].reshape(3,3);tr=np.trace(M);p2=(tr*tr-np.trace(M@M))/2
 z=np.r_[s.y[:3,-1]-y,field(0,y,g,p)[0],3-2*tr+p2]
 if calls%20==0:print('calls',calls,'g',round(g,6),'norm',np.linalg.norm(z),flush=True)
 return z
r=root(residual,x0,tol=1e-8,options={'eps':1e-7,'maxfev':300});print('RESULT',r.success,r.message,'x',r.x,'residual',np.linalg.norm(r.fun),flush=True)
if np.linalg.norm(r.fun)<1e-6:
 y=r.x[:3];T=r.x[3];g=r.x[4]
 def f(t,u):return np.r_[field(t,u[:3],g,p),(jac(u[:3],g,p)@u[3:].reshape(3,3)).ravel()]
 s=solve_ivp(f,(0,T),np.r_[y,np.eye(3).ravel()],method='DOP853',rtol=1e-11,atol=1e-13,max_step=.08,dense_output=True);vals=s.sol(np.linspace(0,T,1500))[:3];ev=np.linalg.eigvals(s.y[3:,-1].reshape(3,3));out={'gamma':float(g),'state':y.tolist(),'period':float(T),'Cmin':float(min(vals[0])),'Cmax':float(max(vals[0])),'Imin':float(min(vals[2])),'Imax':float(max(vals[2])),'multipliers':[[float(x.real),float(x.imag)] for x in ev],'closure':float(np.linalg.norm(s.y[:3,-1]-y)),'residual':float(np.linalg.norm(r.fun)),'note':'Fold candidate: second multiplier near 1; nondegeneracy and transversality have not been verified.'};print(json.dumps(out,indent=2));args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n')
