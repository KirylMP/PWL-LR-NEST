#!/usr/bin/env python3
"""Numerically checked stable/unstable cycles around a first-return fold candidate."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_modes import design

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--fold',type=Path,required=True);ap.add_argument('--verification',type=Path,required=True);ap.add_argument('--reference-fold',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--output-json',type=Path,required=True);a=ap.parse_args();d=json.loads(a.candidate.read_text());fold=json.loads(a.fold.read_text());v=json.loads(a.verification.read_text());ref=json.loads(a.reference_fold.read_text());co=np.array(fold['coefficients']);kn=d['knots'];C=fold['section_C'];rows=[]
 for row in v['branches']:
  if row.get('closure',1)>1e-6:continue
  y0=np.r_[C,row['section_state']];g=row['gamma'];T=row['return_time'];
  def fun(t,y):return design(y,g,kn)[0]@co
  s=solve_ivp(fun,(0,T),y0,method='DOP853',rtol=5e-11,atol=5e-13,max_step=.02,dense_output=True);Y=s.sol(np.linspace(0,T,4000));rows.append({'gamma':g,'kind':'stable' if row['spectral_radius']<1 else 'unstable','period':T,'Cmin':float(min(Y[0])),'Cmax':float(max(Y[0])),'Imin':float(min(Y[2])),'Imax':float(max(Y[2])),'closure_ivp':float(np.linalg.norm(s.y[:,-1]-y0)),'spectral_radius':row['spectral_radius']})
 y0=np.r_[C,fold['section_state']];g=fold['gamma'];T=fold['return']['return_time'];
 def fun(t,y):return design(y,g,kn)[0]@co
 s=solve_ivp(fun,(0,T),y0,method='DOP853',rtol=5e-11,atol=5e-13,max_step=.02,dense_output=True);Y=s.sol(np.linspace(0,T,4000));fr={'gamma':g,'period':T,'Cmin':float(min(Y[0])),'Cmax':float(max(Y[0])),'Imin':float(min(Y[2])),'Imax':float(max(Y[2])),'closure_ivp':float(np.linalg.norm(s.y[:,-1]-y0))};out={'fold':fr,'branches':rows,'reference_fold_candidate':ref,'scope':'Numerical local roots and sample extrema, not formal proof of global continuation or physical invariance.'};a.output_json.parent.mkdir(parents=True,exist_ok=True);a.output_json.write_text(json.dumps(out,indent=2)+'\n')
 fig,axes=plt.subplots(1,3,figsize=(12.5,4),layout='constrained')
 for ax,keys,label in [(axes[0],('Cmin','Cmax'),'C (µM)'),(axes[1],('Imin','Imax'),'IP₃ (µM)'),(axes[2],('period',),'period (s)')]:
  for key in keys:
   for kind,style in [('stable','-'),('unstable','--')]:
    data=sorted((r for r in rows if r['kind']==kind),key=lambda z:z['gamma']);ax.plot([r['gamma'] for r in data]+[g],[r[key] for r in data]+[fr[key]],style,color='#2466a5' if kind=='stable' else '#dd8525',marker='.',ms=5,lw=1.1,label=f'PWA {kind}' if key==keys[0] else None)
   ax.plot(g,fr[key],'D',color='#bf3543',ms=5,label='PWA fold candidate' if key==keys[0] else None);ax.plot(ref['gamma'],ref[key],'D',color='#802eb8',ms=5,label='G-ChI fold candidate' if key==keys[0] else None)
  ax.set_xlabel('Glu (µM)');ax.set_ylabel(label);ax.grid(alpha=.25)
 axes[0].legend(fontsize=7,loc='best');fig.suptitle('Local shooting solutions near the fold: stable and unstable PWA cycles',fontsize=11);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170);print(json.dumps(fr,indent=2))
if __name__=='__main__':main()
