#!/usr/bin/env python3
"""Distinguish sampled transient violations from violations on a detected tail orbit."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_modes import design

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--initial',type=Path,required=True,help='JSON vector [C,h,I]');p.add_argument('--gamma',type=float,required=True);p.add_argument('--duration',type=float,default=1600);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.candidate.read_text());co=np.array(d['coefficients']);kn=d['knots'];y0=np.array(json.loads(a.initial.read_text()));g=a.gamma
 def f(t,y):return design(y,g,kn)[0]@co
 def peak(t,y):return f(t,y)[0]
 peak.direction=-1
 def guard(t,y):return min(y[0]+.05,3-y[0],y[1]+.2,2-y[1],y[2]+.05,4-y[2])
 guard.terminal=True
 s=solve_ivp(f,(0,a.duration),y0,method='DOP853',rtol=3e-8,atol=3e-10,max_step=.3,events=[peak,guard],dense_output=True);t=s.t_events[0];out={'gamma':g,'duration':a.duration,'t_end':float(s.t[-1]),'solver_success':bool(s.success),'min_C_nodes':float(min(s.y[0])),'min_I_nodes':float(min(s.y[2])),'min_h_nodes':float(min(s.y[1])),'max_h_nodes':float(max(s.y[1])),'end':s.y[:,-1].tolist(),'classification':'undetermined','scope':'Numerical nodes and finite sampling; negative values can be transient or sustained. No proof of positivity or complete basin structure.'}
 if s.success and s.t[-1]>=a.duration-1e-6 and len(t)>=4 and t[-2]>.45*a.duration:
  T=t[-1]-t[-2];sample=s.sol(np.linspace(t[-2],t[-1],1500));closure=np.linalg.norm(s.sol(t[-1])-s.sol(t[-2]));amp=np.ptp(sample[0]);periods=np.diff(t[-4:]);out.update(tail_period=float(T),tail_Cmin=float(min(sample[0])),tail_Cmax=float(max(sample[0])),tail_Imin=float(min(sample[2])),tail_Imax=float(max(sample[2])),tail_hmin=float(min(sample[1])),tail_hmax=float(max(sample[1])),tail_closure=float(closure))
  if amp>1e-4 and max(abs(periods-T))/T<.004 and closure<max(1e-5,.003*amp):out['classification']='cycle_with_negative_state' if min(sample[0]) < -1e-8 or min(sample[2]) < -1e-8 or min(sample[1]) < -1e-8 or max(sample[1])>1+1e-8 else 'cycle_physical_in_samples'
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
