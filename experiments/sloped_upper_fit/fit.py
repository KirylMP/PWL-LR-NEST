#!/usr/bin/env python3
"""Exploratory field fit; coordinates C (uM), h, I (uM), seconds."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
KMAP={'AM':.1,'AFM':.075,'FM':.05}
def lr(c,h,i,k):
 q=1.049*(i+.13)/(i+.9434)
 return np.array([(6*(i/(i+.13))**3*(c/(c+.08234))**3*h**3+.11)*(2-1.185*c)-.9*c*c/(c*c+k*k),.2*(q-(q+c)*h)])
def eq(i,k):
 q=1.049*(i+.13)/(i+.9434)
 c=brentq(lambda c:lr(c,q/(q+c),i,k)[0],1e-10,2/1.185)
 return c,q/(q+c)
def upper(i,g):
 c0, i1, width, height=g
 return c0+height*np.clip((i-i1)/width,0,1)
def basis(c,h,i,g):
 c,h,i=np.broadcast_arrays(c,h,i)
 return np.stack([np.ones_like(c),c,h,i,np.maximum(c-g[0],0),np.maximum(c-upper(i,g),0)],axis=-1)
def simulate(fun,i,initial,duration):
 def rhs(t,z):return fun(z[0],z[1],i)
 def bad(t,z):return min(z[0]+.02,2-z[0],z[1]+1,3-z[1])
 bad.terminal=True
 return solve_ivp(rhs,(0,duration),initial,method='DOP853',rtol=2e-7,atol=2e-9,max_step=.3,events=bad,dense_output=True)
def cycles(sol,duration):
 if not sol.success or sol.t[-1]<duration-1e-5:return {'status':'integration_failed_or_left_domain'}
 t=np.linspace(duration*.5,duration,5001); y=sol.sol(t); c=y[0]
 peaks,_=find_peaks(c,prominence=.003,distance=25)
 if len(peaks)<3:return {'status':'no_three_peaks','tail_C_range':float(np.ptp(c))}
 times=t[peaks]; pp=np.diff(times); a,b=peaks[-2:]; period=pp[-1]; amp=np.ptp(c[a:b+1]); peakdiff=abs(c[peaks[-1]]-c[peaks[-2]])
 if amp<.005 or abs(pp[-1]-pp[-2])/max(period,1e-8)>.03 or peakdiff>max(.002,.03*amp):return {'status':'not_converged','period':float(period),'amplitude':float(amp)}
 return {'status':'cycle_detected','period':float(period),'Cmin':float(c[a:b+1].min()),'Cmax':float(c[a:b+1].max()),'peak_difference':float(peakdiff),'period_relative_difference':float(abs(pp[-1]-pp[-2])/period)}
def main():
 (HERE/'results').mkdir(exist_ok=True)
 summaries={}
 for case,k in KMAP.items():
  data=json.loads((HERE/'data'/f'{case.lower()}_validation.json').read_text()); seed=json.loads((HERE/'data'/f'{case.lower()}_parameters.json').read_text())
  items=data['cycles']; train=[v for v in items if v['split']=='train']; held=[v for v in items if v['split']!='train']
  xs=[];ys=[];ws=[]
  for item in train:
   a=item['LR'];i=a['I'];dur=max(300,8*a['period']);sol=simulate(lambda c,h,i:lr(c,h,i,k),i,a['initial'],dur)
   if not sol.success or sol.t[-1]<dur-1e-6:raise RuntimeError('LR integration failed')
   ts=np.linspace(dur-2*a['period'],dur,120); z=sol.sol(ts);ii=np.full(len(ts),i)
   xs.append(np.column_stack((z[0],z[1],ii)));ys.append(lr(z[0],z[1],ii,k).T);ws.append(np.full(len(ts),1/np.sqrt(len(train)*len(ts))))
  # Include the reference rest and equilibrium branch, held-out concentrations excluded.
  lo=min(v['I'] for v in items);hi=max(v['I'] for v in items)
  es=np.linspace(max(0,lo-.12),min(1.3,hi+.12),31)
  for i in es:
   c,h=eq(i,k);xs.append(np.array([[c,h,i]]));ys.append(np.zeros((1,2)));ws.append(np.array([1.5/np.sqrt(len(es))]))
  X=np.concatenate(xs);Y=np.concatenate(ys);W=np.concatenate(ws);scales=np.array([.15,.035])
  def calc(g):
   B=basis(X[:,0],X[:,1],X[:,2],g); B=B*W[:,None]; target=Y*W[:,None]
   coef=np.linalg.lstsq(B,target,rcond=1e-10)[0]
   return ((B@coef-target)/scales).ravel(),coef
  cuts=seed['physical_thresholds_C'];events=seed['fixed_equilibrium_threshold_I'];c0=cuts[0];height=cuts[1]-cuts[0]
  lower=[max(.005,c0*.35),max(0,events[0]-.15),.025,.015];upperb=[min(.5,c0*2.5),min(1.1,events[0]+.15),.5,1.2]
  starts=[np.clip([c0,events[0],w,height],np.array(lower)+1e-5,np.array(upperb)-1e-5) for w in (.07,.15,.3)]
  opts=[least_squares(lambda g:calc(g)[0],s,bounds=(lower,upperb),max_nfev=120,xtol=1e-9,ftol=1e-9) for s in starts]
  opt=min(opts,key=lambda x:np.linalg.norm(x.fun)); g=opt.x;co=calc(g)[1];fun=lambda c,h,i:basis(c,h,i,g)@co
  # Reconstruct frozen physical comparison in its documented N,H basis.
  aa=seed['a'];bb=seed['b'];N=np.array(seed['alpha']);H=np.array(seed['beta']);c1,c2=cuts
  def frozen(c,h,i):
   b=np.array([1,c,i,max(c-c1,0),max(c-c2,0)])
   return np.array([aa*(h-b@N),bb*(b@H-h)])
  rows=[]
  for item in items:
   a=item['LR'];i=a['I'];dur=max(450,14*a['period']);row={'I':i,'split':item['split'],'target':{j:a[j] for j in ('period','Cmin','Cmax')}}
   for name,f in [('new',fun),('frozen',frozen)]:row[name]=cycles(simulate(f,i,a['initial'],dur),dur)
   rows.append(row)
  cmats=[]
  for c in (g[0]-.001,(g[0]+upper(g[1]+g[2],g))/2,upper(g[1]+g[2],g)+.001):
   eps=1e-6;i=g[1]+g[2]+.01;h=.6
   J=np.column_stack(((fun(c+eps,h,i)-fun(c-eps,h,i))/(2*eps),(fun(c,h+eps,i)-fun(c,h-eps,i))/(2*eps)))
   cmats.append({'matrix':J.tolist(),'eigenvalues':[[float(z.real),float(z.imag)] for z in np.linalg.eigvals(J)]})
  result={'case':case,'K':k,'source_commit':'416f56d8959990a1eeaef1a35ae6526f9b1ebeef','geometry':{'C0':g[0],'I1':g[1],'I2':g[1]+g[2],'height':g[3]},'basis':'[1,C,h,I,max(C-C0,0),max(C-upper(I),0)]','coefficient_rows':co.tolist(),'optimizer':{'nfev':opt.nfev,'success':bool(opt.success),'message':opt.message,'weighted_field_objective':float(opt.fun@opt.fun),'geometry_bounds':[lower,upperb]},'matrices':cmats,'validation':rows,'limits':'Exploratory field regression with equilibrium penalty. Validation is forward simulation only, no HLB1 proof, unstable-cycle continuation, fold, event solver, or runtime benchmark.'}
  (HERE/'results'/f'{case.lower()}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');summaries[case]=result
  print(case,'geometry',result['geometry'],'field loss',round(result['optimizer']['weighted_field_objective'],4),'cycles new',sum(v['new']['status']=='cycle_detected' for v in rows),'frozen',sum(v['frozen']['status']=='cycle_detected' for v in rows),flush=True)
 fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
 for row,(case,res) in enumerate(summaries.items()):
  for col,key in enumerate(('period','Cmax')):
   ax=axes[row,col]
   for name,label,color in [('target','LR','black'),('new','new','tab:orange'),('frozen','frozen','tab:blue')]:
    vals=[(v['I'],v[name][key]) for v in res['validation'] if key in v[name] and (name=='target' or v[name]['status']=='cycle_detected')]
    if vals:ax.plot(*zip(*vals),'o-',label=label,color=color,markersize=3)
   for v in res['validation']:
    if v['split']!='train':ax.axvline(v['I'],color='gray',alpha=.18,lw=1)
   ax.set(title=f'{case}: {key}',xlabel='I (µM)',ylabel='seconds' if key=='period' else 'C (µM)');ax.grid(alpha=.2)
   if row==0:ax.legend()
 fig.savefig(HERE/'results'/'comparison.png',dpi=160)
if __name__=='__main__':main()
