#!/usr/bin/env python3
"""Independent adaptive ODE checks; statuses, not branch-existence proofs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from barrier_horizontal import load,f,integrate
P=Path(__file__).parent;ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);ap.add_argument('--duration',type=float,default=800);ap.add_argument('--near-duration',type=float,default=1500);ap.add_argument('--output',type=Path);args=ap.parse_args();m=json.loads(args.candidate.read_text());co,geom=load(str(args.candidate.resolve()));cb=m['barrier']['Cb'];k=m['barrier']['k'];ref=json.loads((P.parent/'gchi_am/output/cycles.json').read_text());records=[]
for idx in np.linspace(0,len(ref)-1,25,dtype=int):
 a=ref[idx];z=integrate(a['gamma'],a['peak'],co,geom,cb,k,duration=args.duration);records.append({'gamma':a['gamma'],'reference':{n:a[n] for n in ('cmin','cmax','period')},'candidate':z});print('gamma',round(a['gamma'],6),z['status'],flush=True)
valid=[r for r in records if r['candidate']['status']=='cycle'];mean={}
for key,rk in [('Cmin','cmin'),('Cmax','cmax'),('period','period')]:mean[key]=float(np.mean([abs(z['candidate'][key]-z['reference'][rk]) for z in valid])) if valid else None
# Equilibrium branch just below H1: enumerate affine cells, including guard.
g=.0115;A0=np.column_stack((co[1],co[2],co[3]));options=[]
for zone in range(4):
 A=A0.copy();b=f(0,[0,0,0],g,co,geom,cb,k)
 if zone==0:A[0,0]-=k
 else:b[0]-=k*cb
 if zone>=2:A[:,0]+=co[7];b-=co[7]*geom[0]
 if zone>=3:A[:,0]+=co[8];b-=co[8]*(geom[0]+geom[3])
 try:y=np.linalg.solve(A,-b)
 except np.linalg.LinAlgError:continue
 C=y[0];ok=[C<=cb,cb<=C<=geom[0],geom[0]<=C<=geom[0]+geom[3],C>=geom[0]+geom[3]][zone]
 if ok and np.all(y>=0):options.append({'state':y.tolist(),'max_real_eig':float(max(np.linalg.eigvals(A).real)),'zone':zone,'residual':float(np.linalg.norm(f(0,y,g,co,geom,cb,k)))})
near=integrate(g,[.22,.70,.36],co,geom,cb,k,duration=args.near_duration)
near['interpretation']='A detected cycle below H1 alongside a stable equilibrium indicates bistability for these initial conditions; no detection does not prove absence.'
anchors=[{'gamma':a['gamma'],'residual':float(np.linalg.norm(f(0,a['state'],a['gamma'],co,geom,cb,k)))} for a in m['fixed_equilibria']]
A=A0.copy();A[0,0]-=k;result={'candidate_path':str(args.candidate),'seed':m.get('seed'),'duration':args.duration,'detected_cycles':len(valid),'total':len(records),'conditional_MAE':mean,'records':records,'below_H1':{'gamma':g,'equilibria':options,'large_initial_condition':[.22,.70,.36],'trajectory':near},'guard_eigenvalues':[[float(v.real),float(v.imag)] for v in np.linalg.eigvals(A)],'anchor_residuals':anchors,'scope':'finite numerical integration, selected initial conditions; not continuation, not a proof of a fold, stability for all ICs, or HLB type'}
out=args.output or args.candidate.parent/'validation.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('saved',out,'cycles',len(valid),'/',len(records),'near-H1',near['status'],'MAE',mean,flush=True)
