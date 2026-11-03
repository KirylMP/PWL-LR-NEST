import json,sys,numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
R=Path(__file__).resolve().parent.parent / 'gchi_modes';sys.path.insert(0,str(R));from fit_modes import design
p=json.load(open(R/'runs/afm-calcium-c1-01/candidate-2.json'));co=np.array(p['coefficients']);kn=np.array(p['knots']);path=Path(str(Path(__file__).resolve().parent / 'am_pwa_unstable.json'));data=json.loads(path.read_text());out=[]
for q in data:
 g=q['gamma'];T=q['period'];x=np.array(q['state'])
 def rhs(t,u):
  y=u[:3];z=np.searchsorted(kn,y[0],side='right');A=co[1:4].T.copy();A[:,0]+=np.sum(co[7:7+z],axis=0) if z else 0;return np.r_[design(y,g,kn)[0]@co,(A@u[3:].reshape(3,3)).ravel()]
 s=solve_ivp(rhs,(0,T),np.r_[x,np.eye(3).ravel()],rtol=2e-10,atol=2e-12,method='DOP853',max_step=.015);ev=np.linalg.eigvals(s.y[3:,-1].reshape(3,3));triv=int(np.argmin(abs(ev-1)));rho=float(np.max(abs(np.delete(ev,triv)))); neutral=float(abs(ev[triv]-1));closure=float(np.linalg.norm(s.y[:3,-1]-x));q.update(rho_variational=rho,neutral_error=neutral,closure_check=closure)
 if rho>1.001 and neutral<1e-4 and closure<1e-6:out.append(q)
 print(round(g,4),round(rho,6),round(neutral,7),flush=True)
path.write_text(json.dumps(out,indent=2)+'\n');print('accepted',len(out),'of',len(data))
