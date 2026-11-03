#!/usr/bin/env python3
"""C-continuous lower PWA barrier added to a prior fixed-anchor candidate."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_horizontal_basis import basis
P=Path(__file__).parent

def load(name='fit_horizontal.json'):
 r=json.loads((P/name).read_text());return np.asarray(r['coefficients']),np.array(list(r['geometry'].values()))

def f(t,y,g,co,geom,cb,k):
 F=basis(y,g,geom)[0]@co
 F[0]+=k*max(cb-y[0],0)
 return F

def integrate(g,init,co,geom,cb,k,duration=800):
 def fun(t,y):return f(t,y,g,co,geom,cb,k)
 def peak(t,y):return fun(t,y)[0]
 peak.direction=-1;peak.terminal=False
 def bound(t,y):return min(y[0]+.05,2-y[0],y[1]+.2,2-y[1],y[2]+.05,3-y[2])
 bound.terminal=True
 s=solve_ivp(fun,(0,duration),init,events=[peak,bound],method='DOP853',rtol=3e-7,atol=3e-9,max_step=.4,dense_output=True)
 if not s.success or s.t[-1]<duration-1e-6:return {'status':'domain_or_integration_failure','t_end':float(s.t[-1]),'end':s.y[:,-1].tolist()}
 t=s.t_events[0];t=t[t>duration*.5]
 if len(t)<4:return {'status':'no_cycle_detected','end':s.y[:,-1].tolist()}
 d=np.diff(t[-4:]);z=s.sol(np.linspace(t[-2],t[-1],500));amp=np.ptp(z[0]);closure=np.linalg.norm(s.sol(t[-1])-s.sol(t[-2]));
 if amp<1e-4 or np.max(abs(d-d[-1]))/d[-1]>.004 or closure>max(1e-5,.003*amp):return {'status':'not_converged','end':s.y[:,-1].tolist()}
 return {'status':'cycle','Cmin':float(z[0].min()),'Cmax':float(z[0].max()),'period':float(d[-1]),'closure':float(closure),'end':s.y[:,-1].tolist()}
