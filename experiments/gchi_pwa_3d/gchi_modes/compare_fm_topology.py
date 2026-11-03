#!/usr/bin/env python3
"""Compare independently integrated candidate trajectories; no unstable cycle continuation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_modes import design

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--reference-scan',type=Path,required=True);p.add_argument('--reference-fold',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[0,3,28,31]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();ref=json.loads(a.reference.read_text());scan=json.loads(a.reference_scan.read_text());fold=json.loads(a.reference_fold.read_text());rc=[x for x in ref['cycles'] if x['status']=='cycle'];rc += [dict(x['result'],gamma=x['gamma']) for x in scan['reference_low'] if x['result']['status']=='cycle'];rc += [dict(gamma=fold['gamma'],Cmin=fold['Cmin'],Cmax=fold['Cmax'],Imin=fold['Imin'],Imax=fold['Imax'],period=fold['period'])];rc.sort(key=lambda x:x['gamma']);re=[]
 for row in ref['equilibria']:
  if row['gamma']>0:
   for q in row['roots']:re.append([row['gamma'],*q['state'],q['max_real']])
 re=np.array(re);fig,axes=plt.subplots(len(a.seeds),3,figsize=(13.2,3.15*len(a.seeds)),layout='constrained',squeeze=False)
 for axrow,seed in zip(axes,a.seeds):
  can=json.loads((a.run/f'candidate-{seed}.json').read_text());val=json.loads((a.run/f'validation-{seed}.json').read_text());on=json.loads((a.run/f'onset-{seed}.json').read_text());co=np.array(can['coefficients']);kn=can['knots'];pc=[dict(x['candidate'],gamma=x['gamma']) for x in val['records'] if x['candidate']['status']=='cycle'];pc += [dict(x['candidate'],gamma=x['gamma']) for x in on['records'] if x['initial_type']=='previous_pwa_cycle' and x['candidate']['status']=='cycle'];pc.sort(key=lambda x:x['gamma']);eq=[]
  for g in np.geomspace(.012,30,350):
   for j,C in enumerate([kn[0]/2,(kn[0]+kn[1])/2,(kn[1]+kn[2])/2,kn[2]+.05]):
    A=co[1:4].T.copy()
    for k in range(j):A[:,0]+=co[7+k]
    y0=np.array([C,.8,.4])
    try:y=y0-np.linalg.solve(A,design(y0,g,kn)[0]@co)
    except np.linalg.LinAlgError:continue
    if (j==0 or y[0]>=kn[j-1]) and (j==3 or y[0]<=kn[j]) and y[0]>=0 and y[2]>=0 and 0<=y[1]<=1:eq.append([g,*y,max(np.linalg.eigvals(A).real)])
  eq=np.array(eq)
  for k,(ax,col,keys,label) in enumerate([(axrow[0],1,('Cmin','Cmax'),'C (µM)'),(axrow[1],3,('Imin','Imax'),'IP₃ (µM)')]):
   for stable in [True,False]:
    mask=(re[:,4]<0)==stable;ax.semilogx(re[mask,0],re[mask,col],'.',color='#8da1b1' if stable else '#d6b17b',ms=2.7,alpha=.55,label=('G-ChI eq. stable' if stable else 'G-ChI eq. unstable') if seed==a.seeds[0] else None)
    if len(eq):
     mask=(eq[:,4]<0)==stable;ax.semilogx(eq[mask,0],eq[mask,col],'.',color='#236aa0' if stable else '#bd5c9c',ms=2.5,alpha=.75,label=('PWA eq. stable' if stable else 'PWA eq. unstable') if seed==a.seeds[0] else None)
   for n,key in enumerate(keys):
    ax.semilogx([x['gamma'] for x in rc],[x[key] for x in rc],'-',color='#26834a',lw=1.2,label='G-ChI cycle extrema' if n==0 and seed==a.seeds[0] else None)
    ax.semilogx([x['gamma'] for x in pc],[x[key] for x in pc],'x',color='#b72727',ms=4,label='PWA detected cycles' if n==0 and seed==a.seeds[0] else None)
    ax.semilogx([fold['gamma']],[fold[key]],'D',color='#8e37be',ms=3.8,label='G-ChI fold candidate' if n==0 and seed==a.seeds[0] else None)
   ax.set_ylabel(f'Seed {seed}: {label}' if k==0 else label);ax.set_xlim(.012,30);ax.grid(alpha=.2)
  ax=axrow[2];ax.loglog([x['gamma'] for x in rc],[x['period'] for x in rc],'-',color='#26834a',lw=1.2,label='G-ChI cycles' if seed==a.seeds[0] else None);ax.loglog([x['gamma'] for x in pc],[x['period'] for x in pc],'x',color='#b72727',ms=4,label='PWA detected cycles' if seed==a.seeds[0] else None);ax.loglog([fold['gamma']],[fold['period']],'D',color='#8e37be',ms=4,label='G-ChI fold candidate' if seed==a.seeds[0] else None);ax.set_xlim(.012,30);ax.set_ylabel('period (s), log');ax.grid(alpha=.2)
 for ax in axes[-1]:ax.set_xlabel('Glu (µM)')
 axes[0,0].legend(fontsize=6,loc='upper left',ncol=1);axes[0,2].legend(fontsize=6);fig.suptitle('FM topology search: sampled equilibria and detected cycles; fold marker is G-ChI only',fontsize=11);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170)
if __name__=='__main__':main()
