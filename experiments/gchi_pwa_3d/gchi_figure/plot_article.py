import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parent.parent / 'gchi_modes';sys.path.insert(0,str(R));from fit_modes import design
OUT=Path(__file__).resolve().parent
S={'AM':('AFM','runs/afm-calcium-c1-01/candidate-2.json','runs/afm-calcium-c1-01/validation-2.json',4.2),'FM':('FM','runs/fm-topology-free-c1-01/candidate-3.json','runs/fm-topology-free-c1-01/validation-3.json',23.)}
ref_unstable_am=[q for q in json.loads((R.parent/'gchi_am/output/continuation.json').read_text()) if q['rho']>1.001 and q['Cmax']-q['Cmin']>1e-4 and q['residual']<1e-7]
ref_unstable_fm=json.loads((OUT/'fm_reference_unstable.json').read_text())
am_pwa_unstable=json.loads((OUT/'am_pwa_unstable.json').read_text())
for variable in ('C','I'):
 fig,axes=plt.subplots(2,2,figsize=(10.5,6.8),sharex='col',sharey='col',layout='constrained');keymin,keymax=variable+'min',variable+'max'; ix=0 if variable=='C' else 2
 for col,(label,(mode,cp,vp,xmax)) in enumerate(S.items()):
  d=json.loads((R/cp).read_text());co=np.array(d['coefficients']);kn=np.array(d['knots']);ref=json.loads((R/'results'/mode/'reference.json').read_text());v=json.loads((R/vp).read_text()); gammas=np.linspace(0,xmax,900)
  for j in range(4):
   states=[];st=[]
   for g in gammas:
    A=co[1:4].T.copy();A[:,0]+=np.sum(co[7:7+j],axis=0) if j else 0;b=design(np.zeros(3),g,kn)[0]@co
    if j:b-=sum(co[7+i]*kn[i] for i in range(j))
    try:y=np.linalg.solve(A,-b)
    except np.linalg.LinAlgError:y=np.full(3,np.nan)
    lo=-np.inf if j==0 else kn[j-1];hi=np.inf if j==3 else kn[j];ok=lo-1e-9<=y[0]<=hi+1e-9 and y[0]>=0 and y[2]>=0 and 0<=y[1]<=1
    states.append(y if ok else np.full(3,np.nan));st.append(ok and np.max(np.real(np.linalg.eigvals(A))) < -1e-8)
   states=np.array(states);st=np.array(st)
   for stable,color,ls in ((True,'#245f9b','-'),(False,'#bf434a','--')):axes[1,col].plot(gammas,np.where(st==stable,states[:,ix],np.nan),color=color,ls=ls,lw=1.25)
  # The reference root search is sampled and can contain multiple equilibria.
  for item in ref['equilibria']:
   for q in item['roots']:
    stable=q['max_real'] < -1e-8;axes[0,col].scatter(item['gamma'],q['state'][ix],s=8,color='#245f9b' if stable else '#bf434a',marker='.' if stable else 'x',linewidth=.55,zorder=2)
  for source,ax,kind in [(ref['cycles'],axes[0,col],'reference'),(v['records'],axes[1,col],'pwa')]:
   data=[q for q in source if q.get('status')=='cycle'] if kind=='reference' else [dict(gamma=q['gamma'],**q['candidate']) for q in source if q['candidate'].get('status')=='cycle']
   for key in (keymin,keymax):ax.scatter([q['gamma'] for q in data],[q[key] for q in data],s=11,c='#222222',zorder=4)
  if label=='AM':
   for key in (keymin,keymax):axes[1,col].scatter([q['gamma'] for q in am_pwa_unstable],[q[key] for q in am_pwa_unstable],s=17,facecolors='white',edgecolors='#a32839',linewidths=.8,zorder=5)
  unstable=ref_unstable_am if label=='AM' else ref_unstable_fm
  for key in (keymin,keymax):axes[0,col].scatter([q['gamma'] for q in unstable],[q[key] for q in unstable],s=17,facecolors='white',edgecolors='#a32839',linewidths=.8,zorder=5)
  fold=json.loads((R.parent/'gchi_am/output/fold.json').read_text()) if label=='AM' else json.loads((R/'analysis/fm-constrained-01/reference-fold-candidate.json').read_text())
  for key in (keymin,keymax):axes[0,col].scatter(fold['gamma'],fold[key],marker='D',s=23,color='#7354a4',zorder=6)
  axes[0,col].set_title(label+(' (G-ChI AM parameters)' if label=='AM' else ' (G-ChI FM parameters)'),fontsize=10)
  axes[1,col].set_xlabel(r'Glutamate $\gamma$ ($\mu$M)')
  for row in (0,1):axes[row,col].set_xlim(0,xmax);axes[row,col].set_ylim(bottom=0);axes[row,col].grid(alpha=.16)
 ylabel=r'$C$ ($\mu$M)' if variable=='C' else r'$I$ ($\mu$M)'
 axes[0,0].set_ylabel('G-ChI\n'+ylabel);axes[1,0].set_ylabel('PWA\n'+ylabel)
 fig.legend(handles=[Line2D([],[],color='#245f9b',label='stable equilibrium'),Line2D([],[],color='#bf434a',ls='--',label='unstable equilibrium'),Line2D([],[],marker='o',ls='',color='#222',ms=4,label='stable cycle'),Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor='#a32839',ms=5,label='unstable cycle'),Line2D([],[],marker='D',ls='',color='#7354a4',ms=4,label='fold candidate')],loc='outside lower center',ncol=5,frameon=False,fontsize=8)
 fig.suptitle(('Calcium' if variable=='C' else 'IP₃')+' equilibria and cycle extrema',fontsize=12)
 for fmt in ('png','pdf','svg'):fig.savefig(OUT/f'gchi_{variable}_article.{fmt}',dpi=220,bbox_inches='tight')
 plt.close(fig)
print('reference unstable cycle samples AM',len(ref_unstable_am),'FM',len(ref_unstable_fm))
