#!/usr/bin/env python3
"""FM equilibrium points, detected cycles and reference fold candidate."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_modes import design
from explore_modes import P,roots,jac

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--validation',type=Path,required=True);ap.add_argument('--scans',type=Path,required=True);ap.add_argument('--reference-high',type=Path,required=True);ap.add_argument('--reference-fold',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--xmin',type=float,default=.001);ap.add_argument('--xmax',type=float,default=10000);a=ap.parse_args()
 ref=json.loads(a.reference.read_text());can=json.loads(a.candidate.read_text());v=json.loads(a.validation.read_text());scans=json.loads(a.scans.read_text());high=json.loads(a.reference_high.read_text());fold=json.loads(a.reference_fold.read_text());co=np.array(can['coefficients']);kn=np.array(can['knots']);M=[co[1:4,:].T.copy()]
 for j in range(3):
  n=M[-1].copy();n[:,0]+=co[7+j];M.append(n)
 re=[]
 for row in ref['equilibria']:
  for q in row['roots']:
   if row['gamma']>0:re.append((row['gamma'],*q['state'],q['max_real']))
 guess=np.array([.54,.56,1.45])
 for g in np.geomspace(20.01,10000,100):
  rr=roots(g,P['FM'],[guess])
  if rr:
   guess=rr[-1];re.append((g,*guess,max(np.linalg.eigvals(jac(guess,g,P['FM'])).real)))
 pe=[]
 for g in np.geomspace(.001,10000,700):
  for j,c in enumerate([kn[0]/2,(kn[0]+kn[1])/2,(kn[1]+kn[2])/2,kn[2]+.05]):
   y0=np.array([c,.8,.4]);A=M[j]
   try:y=y0-np.linalg.solve(A,design(y0,g,kn)[0]@co)
   except np.linalg.LinAlgError:continue
   if (j==0 or y[0]>=kn[j-1]) and (j==3 or y[0]<=kn[j]) and y[0]>=0 and y[2]>=0 and 0<=y[1]<=1:pe.append((g,*y,max(np.linalg.eigvals(A).real)))
 re=np.array(re);pe=np.array(pe);rc=[q for q in ref['cycles'] if q['status']=='cycle'];rc += [dict(x['result'],gamma=x['gamma']) for x in scans['reference_low'] if x['result']['status']=='cycle'];rc += [q for q in high if q['status']=='cycle'];rc.sort(key=lambda z:z['gamma'])
 pc=[dict(x['candidate'],gamma=x['gamma']) for x in v['records'] if x['candidate']['status']=='cycle'];pc += [dict(x['result'],gamma=x['gamma']) for key in ('PWA_low','PWA_high') for x in scans[key] if x['result']['status']=='cycle'];pc.sort(key=lambda z:z['gamma'])
 fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
 for ax,col,ext,ylabel in [(axes[0],1,('Cmin','Cmax'),'C (µM)'),(axes[1],3,('Imin','Imax'),'IP₃ (µM)')]:
  for stable in [True,False]:
   mask=(re[:,4]<0)==stable;ax.semilogx(re[mask,0],re[mask,col],'.',color='#25588f' if stable else '#e08a25',ms=3,label='G-ChI equilibrium '+('stable' if stable else 'unstable'))
   if len(pe):
    mask=(pe[:,4]<0)==stable;ax.semilogx(pe[mask,0],pe[mask,col],'.',color='#7bbbd1' if stable else '#d58ab3',ms=2.2,label='PWA equilibrium '+('stable' if stable else 'unstable'))
  for j,key in enumerate(ext):
   ax.semilogx([r['gamma'] for r in rc],[r[key] for r in rc],'-',color='#3d8753',lw=1.3,label='G-ChI cycle extrema' if j==0 else None)
   ax.semilogx([r['gamma'] for r in pc],[r[key] for r in pc],'x',color='#b32e2e',ms=4,label='PWA detected cycles' if j==0 else None)
   ax.semilogx([fold['gamma']],[fold[key]],'D',color='#862dd3',ms=4,label='G-ChI fold candidate' if j==0 else None)
  ax.set_ylabel(ylabel);ax.set_xlabel('Glu (µM)');ax.grid(alpha=.2)
 axes[2].loglog([r['gamma'] for r in rc],[r['period'] for r in rc],'-',color='#3d8753',lw=1.3,label='G-ChI cycles');axes[2].loglog([r['gamma'] for r in pc],[r['period'] for r in pc],'x',color='#b32e2e',ms=4,label='PWA detected cycles');axes[2].loglog([fold['gamma']],[fold['period']],'D',color='#862dd3',ms=4,label='G-ChI fold candidate');axes[2].set_ylabel('period (s), log scale');axes[2].set_xlabel('Glu (µM)');axes[2].grid(alpha=.2)
 [ax.set_xlim(a.xmin,a.xmax) for ax in axes];axes[0].legend(fontsize=6,loc='upper left');axes[2].legend(fontsize=7);fig.suptitle('FM: sampled equilibria and cycles; reference fold candidate is provisional',fontsize=11);a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=175)
if __name__=='__main__':main()
