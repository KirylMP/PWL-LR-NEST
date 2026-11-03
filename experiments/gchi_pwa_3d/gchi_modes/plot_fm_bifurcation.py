#!/usr/bin/env python3
"""Sampled equilibrium branches and detected cycles for reference G-ChI and PWA."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_modes import design

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--validation',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();ref=json.loads(a.reference.read_text());can=json.loads(a.candidate.read_text());v=json.loads(a.validation.read_text());co=np.array(can['coefficients']);kn=np.array(can['knots']);A0=co[1:4,:].T.copy();mats=[A0.copy()]
 for j in range(3):
  m=mats[-1].copy();m[:,0]+=co[7+j];mats.append(m)
 rdata=[]
 for x in ref['equilibria']:
  if x['gamma']>0:
   for root in x['roots']:rdata.append((x['gamma'],*root['state'],root['max_real']))
 pwa=[]
 for g in np.geomspace(.001,20,450):
  for j,c in enumerate([kn[0]/2,(kn[0]+kn[1])/2,(kn[1]+kn[2])/2,kn[2]+.05]):
   y0=np.array([c,.8,.4]);A=mats[j]
   try:y=y0-np.linalg.solve(A,design(y0,g,kn)[0]@co)
   except np.linalg.LinAlgError:continue
   if (j==0 or y[0]>=kn[j-1]) and (j==3 or y[0]<=kn[j]) and y[0]>=0 and y[2]>=0 and 0<=y[1]<=1:pwa.append((g,*y,max(np.linalg.eigvals(A).real)))
 rdata=np.array(rdata);pwa=np.array(pwa);cy=[x for x in ref['cycles'] if x['status']=='cycle'];pc=[x for x in v['records'] if x['candidate']['status']=='cycle'];fig,axs=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
 for ax,col,extrema,ylabel in [(axs[0],1,('Cmin','Cmax'),'C (µM)'),(axs[1],3,('Imin','Imax'),'IP₃ (µM)')]:
  for stable in [True,False]:
   mask=(rdata[:,4]<0)==stable;ax.semilogx(rdata[mask,0],rdata[mask,col],'.',color='#275c9c' if stable else '#e08a25',ms=4,alpha=.75,label='G-ChI equilibrium '+('stable' if stable else 'unstable'))
   if len(pwa):
    mask=(pwa[:,4]<0)==stable;ax.semilogx(pwa[mask,0],pwa[mask,col],'.',color='#79bad1' if stable else '#d98bb9',ms=2.6,alpha=.8,label='PWA equilibrium '+('stable' if stable else 'unstable'))
  for j,key in enumerate(extrema):
   ax.semilogx([x['gamma'] for x in cy],[x[key] for x in cy],color='#3d8753',lw=1.3,label='G-ChI cycle extrema' if j==0 else None)
   ax.semilogx([x['gamma'] for x in pc],[x['candidate'][key] for x in pc],'x',color='#b32e2e',ms=5,label='PWA detected cycle extrema' if j==0 else None)
  ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.set_xlabel('Glu (µM)')
 axs[2].semilogx([x['gamma'] for x in cy],[x['period'] for x in cy],color='#3d8753',lw=1.5,label='G-ChI cycles');axs[2].semilogx([x['gamma'] for x in pc],[x['candidate']['period'] for x in pc],'x',color='#b32e2e',ms=5,label='PWA detected cycles');axs[2].set_ylabel('period (s)');axs[2].set_xlabel('Glu (µM)');axs[2].grid(alpha=.2)
 axs[0].legend(fontsize=6,loc='upper left');axs[2].legend(fontsize=7);fig.suptitle('FM: sampled equilibria and forward-detected cycles — candidate remains provisional',fontsize=11);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=175)
if __name__=='__main__':main()
