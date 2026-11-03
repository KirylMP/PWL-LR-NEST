#!/usr/bin/env python3
"""Validate variable-guard PWA and probe several basins below H1."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from barrier_horizontal import load,f
P=Path(__file__).parent
ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);ap.add_argument('--duration',type=float,default=800);ap.add_argument('--probe-duration',type=float,default=1200);ap.add_argument('--output',type=Path);args=ap.parse_args()
out=args.output or args.candidate.parent/'validation.json'
subprocess.run([sys.executable,str(P/'validate_candidate.py'),str(args.candidate),'--duration',str(args.duration),'--near-duration',str(args.probe_duration),'--output',str(out)],check=True)
r=json.loads(out.read_text());m=json.loads(args.candidate.read_text());co,geom=load(str(args.candidate.resolve()));cb=m['barrier']['Cb'];k=m['barrier']['k'];probes=[]
for gamma in [.0115,.0125,.0129]:
 eq=root(lambda y:f(0,y,gamma,co,geom,cb,k),[.15,.72,.34],tol=1e-10)
 if np.linalg.norm(eq.fun)>1e-8 or np.min(eq.x)<0:
  probes.append({'gamma':gamma,'status':'equilibrium_not_resolved','residual':float(np.linalg.norm(eq.fun))});continue
 for label,ini in [('standard',[.22,.7,.36]),('large',[.32,.65,.45]),('low',[.11,.78,.25])]:
  def rhs(t,y):return f(t,y,gamma,co,geom,cb,k)
  def peak(t,y):return rhs(t,y)[0]
  peak.direction=-1;peak.terminal=False
  def bound(t,y):return min(y[0]+.05,2-y[0],y[1]+.2,2-y[1],y[2]+.05,3-y[2])
  bound.terminal=True
  sol=solve_ivp(rhs,(0,args.probe_duration),ini,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.3,events=[peak,bound],dense_output=True)
  if not sol.success or sol.t[-1]<args.probe_duration-1e-5:
   probes.append({'gamma':gamma,'initial':label,'status':'domain_or_integration_failure','t_end':float(sol.t[-1]),'end_state':sol.y[:,-1].tolist(),'min_C_at_solver_nodes':float(np.min(sol.y[0]))});continue
  tt=np.linspace(args.probe_duration*.85,args.probe_duration,800);tail=sol.sol(tt);span=float(np.ptp(tail[0]));dist=float(np.linalg.norm(sol.y[:,-1]-eq.x));status='undetermined';period=None
  ts=sol.t_events[0];ts=ts[ts>args.probe_duration*.5]
  if len(ts)>=4:
   ds=np.diff(ts[-4:]);period=float(ds[-1]);closure=float(np.linalg.norm(sol.sol(ts[-1])-sol.sol(ts[-2])))
   if span>.001 and max(abs(ds-ds[-1]))/ds[-1]<.004 and closure<max(1e-5,.003*span):status='cycle'
  if dist<1e-5 and span<1e-6:status='near_equilibrium'
  min_C=float(np.min(sol.y[0]));
  if min_C < -1e-6:status='negative_calcium'
  probes.append({'min_C_at_solver_nodes':min_C,'gamma':gamma,'initial':label,'status':status,'end_distance':dist,'tail_C_range':span,'period':period,'equilibrium':eq.x.tolist()});print('probe',gamma,label,status,'tail_C_range',span,'end_distance',dist,flush=True)
r['near_h1_sweep']={'duration':args.probe_duration,'probes':probes,'scope':'finite multiple-IC tests; non-detection does not prove absence of cycles'};out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print('saved sweep',out,flush=True)
