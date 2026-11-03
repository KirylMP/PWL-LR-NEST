#!/usr/bin/env python3
"""Visualize independently relaxed trajectories at the onset; phases are not aligned."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_modes import design
from explore_modes import field,P

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--reference-scan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=1600);a=ap.parse_args();g=.0203;scan=json.loads(a.reference_scan.read_text());init=next(x['result']['peak'] for x in scan['reference_low'] if abs(x['gamma']-g)<1e-8);spec=[('G-ChI reference',None,init)]
 for seed in [0,3]:
  can=json.loads((a.run/f'candidate-{seed}.json').read_text());on=json.loads((a.run/f'onset-{seed}.json').read_text());y0=next(x['initial'] for x in on['records'] if abs(x['gamma']-g)<1e-8 and x['initial_type']=='previous_pwa_cycle');spec.append((f'PWA seed {seed}',can,y0))
 fig,ax=plt.subplots(3,2,figsize=(11,6.9),sharex=True,layout='constrained');ts=np.linspace(a.duration-350,a.duration,3501)
 for j,(title,can,y0) in enumerate(spec):
  if can is None:fun=lambda t,y:field(t,y,g,P['FM'])
  else:
   co=np.array(can['coefficients']);kn=can['knots'];fun=lambda t,y,co=co,kn=kn:design(y,g,kn)[0]@co
  s=solve_ivp(fun,(0,a.duration),y0,t_eval=ts,method='DOP853',rtol=3e-8,atol=3e-10,max_step=.3)
  if not s.success or len(s.t)!=len(ts):raise RuntimeError(f'{title}: integration failed')
  for k in range(2):
   values=s.y[[0,2][k]];ax[j,k].plot(ts-ts[0],values,lw=1.2,color='#26834a' if j==0 else '#b72727');ax[j,k].axhline(0,color='gray',lw=.8);ax[j,k].grid(alpha=.2);ax[j,k].set_ylabel(f'{title}\n'+('C (µM)' if k==0 else 'IP₃ (µM)'))
 for x in ax[-1]:x.set_xlabel('time in displayed window (s)')
 fig.suptitle('Glu = 0.0203 µM: relaxed trajectories; phases are independent',fontsize=11);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170)
if __name__=='__main__':main()
