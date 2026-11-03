#!/usr/bin/env python3
"""Continuous PWA trial for 3D G-ChI. Fit RHS on reference equilibria and cycles."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from reference_gchi import field
ROOT=Path(__file__).parent
ref=ROOT.parent/'gchi_am'/'output'
cycles=json.loads((ref/'cycles.json').read_text())
eq=np.genfromtxt(ref/'equilibria.csv',delimiter=',',names=True)

def basis(y,g,geom):
 y=np.atleast_2d(y);g=np.broadcast_to(g,len(y));C,h,I=y.T
 c0,i1,width,height=geom;top=c0+height*np.clip((I-i1)/width,0,1)
 # input nonlinearities depend only on constant parameter g, not state
 v=np.maximum(g,0)**.7
 return np.column_stack([np.ones(len(y)),C,h,I,v/(v+.02**.7),v/(v+.5**.7),v/(v+3**.7),np.maximum(C-c0,0),np.maximum(C-top,0)])

def generate():
 xs=[];gs=[];ys=[];ws=[]
 for idx in np.linspace(1,len(cycles)-2,19,dtype=int):
  r=cycles[idx];g=r['gamma'];T=r['period'];ts=np.linspace(0,T,96,endpoint=False)
  sol=solve_ivp(field,(0,T),r['peak'],args=(g,),t_eval=ts,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.2)
  x=sol.y.T;xs.extend(x);gs.extend([g]*len(x));ys.extend([field(0,z,g) for z in x]);ws.extend([1/np.sqrt(19*len(x))]*len(x))
 ids=np.linspace(25,130,80,dtype=int)
 for z in eq[ids]:
  x=np.array([z['C'],z['h'],z['I']]);xs.append(x);gs.append(z['gamma']);ys.append(np.zeros(3));ws.append(2/np.sqrt(len(ids)))
 return np.array(xs),np.array(gs),np.array(ys),np.array(ws)

def main():
 X,g,Y,w=generate();scales=np.array([.12,.025,.025])
 def evaluate(geom,ret=False):
  A=basis(X,g,geom)*w[:,None];b=Y*w[:,None];co=np.linalg.lstsq(A,b,rcond=1e-10)[0];err=(A@co-b)/scales
  return (err.ravel(),co) if ret else err.ravel()
 lower=[.07,.1,.04,.02];upper=[.28,.7,.8,.5]
 starts=[[.155,.355,.28,.17],[.12,.3,.45,.2],[.18,.4,.2,.13],[.14,.25,.5,.25]]
 opts=[least_squares(evaluate,x,bounds=(lower,upper),max_nfev=200,ftol=2e-8) for x in starts]
 opt=min(opts,key=lambda x:x.cost);res,co=evaluate(opt.x,True)
 out={'geometry':dict(zip(('C0','I1','width','height'),opt.x.tolist())),'coefficients':co.tolist(),'basis':'1,C,h,I,Hill(gamma;0.02),Hill(gamma;0.5),Hill(gamma;3),positive(C-C0),positive(C-upper(I))','input_exponent':.7,'field_fit_rmse_weighted_scaled':float(np.linalg.norm(res)),'starts':[{'cost':float(z.cost),'success':bool(z.success),'nfev':z.nfev} for z in opts],'limits':'Field regression only; no bifurcation or cycle guarantees.'}
 (ROOT/'fit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
