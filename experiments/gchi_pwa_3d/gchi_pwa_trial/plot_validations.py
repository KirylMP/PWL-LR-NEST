#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;ap=argparse.ArgumentParser();ap.add_argument('old',type=Path);ap.add_argument('new',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();ref=json.loads((P.parent/'gchi_am/output/cycles.json').read_text());old=json.loads(a.old.read_text());new=json.loads(a.new.read_text());fig,axs=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
for j,axis in enumerate(axs[:2]):
 for k in ('cmin','cmax'):axis.plot([z['gamma'] for z in ref],[z[k] for z in ref],'.',color='#888',ms=3,label='G-ChI' if k=='cmin' else None)
 for data,col,label in [(old,'#dc8b20','previous PWA'),(new,'#2875a8','updated PWA')]:
  good=[r for r in data['records'] if r['candidate']['status']=='cycle']
  for k in ('Cmin','Cmax'):axis.plot([z['gamma'] for z in good],[z['candidate'][k] for z in good],'.',color=col,ms=5,label=label if k=='Cmin' else None)
  probe=data['below_H1'];v=probe['trajectory'];
  if v['status']=='cycle':axis.plot([probe['gamma']]*2,[v['Cmin'],v['Cmax']],'x',color=col,ms=5,label=label+' cycle below H1')
 axis.set_xlabel('glutamate (µM)');axis.set_ylabel('C (µM)');axis.grid(alpha=.2);axis.axvline(.012990740120612475,color='#aaa',ls=':',lw=.8)
axs[0].set_xscale('log');axs[0].set_xlim(.008,3);axs[0].set_title('Cycle extrema');axs[0].legend(fontsize=7)
axs[1].set_xlim(.0105,.035);axs[1].set_ylim(.07,.29);axs[1].set_title('Near first transition')
axs[2].plot([z['gamma'] for z in ref],[z['period'] for z in ref],'.',color='#888',ms=3,label='G-ChI')
for data,col,label in [(old,'#dc8b20','previous PWA'),(new,'#2875a8','updated PWA')]:
 good=[r for r in data['records'] if r['candidate']['status']=='cycle'];axs[2].plot([z['gamma'] for z in good],[z['candidate']['period'] for z in good],'.',color=col,ms=5,label=label)
axs[2].set_xscale('log');axs[2].set_xlim(.008,3);axs[2].set_xlabel('glutamate (µM)');axs[2].set_ylabel('period (s)');axs[2].set_title('Period');axs[2].grid(alpha=.2);axs[2].legend(fontsize=8)
fig.suptitle('PWA fit with calcium threshold 0.1 µM');a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=175)
