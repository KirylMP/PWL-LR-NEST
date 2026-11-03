#!/usr/bin/env python3
"""Independent adaptive integration, held-out gamma points and reference comparison."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_modes import design

def simulate(g,y0,co,knots,duration):
 def f(t,y):return design(y,g,knots)[0]@co
 def peak(t,y):return f(t,y)[0]
 peak.direction=-1
 def bound(t,y):return min(y[0]+.05,3-y[0],y[1]+.2,2-y[1],y[2]+.05,4-y[2])
 bound.terminal=True
 s=solve_ivp(f,(0,duration),y0,method='DOP853',events=[peak,bound],dense_output=True,rtol=3e-8,atol=3e-10,max_step=.3)
 out={'status':'undetermined','min_C_solver_nodes':float(min(s.y[0])),'min_I_solver_nodes':float(min(s.y[2])),'min_h_solver_nodes':float(min(s.y[1])),'max_h_solver_nodes':float(max(s.y[1])),'t_end':float(s.t[-1]),'end':s.y[:,-1].tolist()}
 violations=[]
 if np.min(s.y[0]) < -1e-8:violations.append('negative_calcium')
 if np.min(s.y[2]) < -1e-8:violations.append('negative_ip3')
 if np.min(s.y[1]) < -1e-8 or np.max(s.y[1]) > 1+1e-8:violations.append('h_outside_0_1')
 if violations:out['physical_violations']=violations;out['status']=violations[0];return out
 if not s.success or s.t[-1]<duration-1e-5:out['status']='domain_or_integration_failure';return out
 ts=s.t_events[0];ts=ts[ts>duration*.45]
 if len(ts)<4:return out
 delta=np.diff(ts[-4:]);ys=s.sol(np.linspace(ts[-2],ts[-1],300));amp=np.ptp(ys[0]);closure=np.linalg.norm(s.sol(ts[-1])-s.sol(ts[-2]));out.update(Cmin=float(min(ys[0])),Cmax=float(max(ys[0])),Imin=float(min(ys[2])),Imax=float(max(ys[2])),period=float(delta[-1]),closure=float(closure))
 if np.min(ys[0]) < -1e-8:out['status']='negative_calcium'
 elif np.min(ys[2]) < -1e-8:out['status']='negative_ip3'
 elif np.min(ys[1]) < -1e-8 or np.max(ys[1]) > 1+1e-8:out['status']='h_outside_0_1'
 elif amp>1e-4 and max(abs(delta-delta[-1]))/delta[-1]<.004 and closure<max(1e-5,.003*amp):out['status']='cycle'
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=1000);a=ap.parse_args();ref=json.loads(a.reference.read_text());can=json.loads(a.candidate.read_text());co=np.array(can['coefficients']);knots=np.array(can['knots']);valid=[];gs=set(can.get('training_gammas',[]));good=[r for r in ref['cycles'] if r['status']=='cycle' and r['gamma'] not in gs];idx=np.unique(np.linspace(0,len(good)-1,min(12,len(good)),dtype=int))
 for i in idx:
  r=good[i];res=simulate(r['gamma'],r['peak'],co,knots,a.duration);valid.append({'gamma':r['gamma'],'reference':{k:r[k] for k in ('Cmin','Cmax','Imin','Imax','period')},'candidate':res});print(can['mode'],round(r['gamma'],5),res['status'],flush=True)
 ok=[r for r in valid if r['candidate']['status']=='cycle'];mae={k:float(np.mean([abs(r['reference'][k]-r['candidate'][k]) for r in ok])) if ok else None for k in ('Cmin','Cmax','Imin','Imax','period')}
 zero=ref['equilibria'][0]['roots'];eigs=[]
 for z in zero:
  y=np.array(z['state']);eps=1e-7;J=np.column_stack([(design(y+np.eye(3)[i]*eps,0,knots)[0]@co-design(y-np.eye(3)[i]*eps,0,knots)[0]@co)/(2*eps) for i in range(3)]);eigs.append({'state':z['state'],'ref_max_real':z['max_real'],'pwa_max_real':float(max(np.linalg.eigvals(J).real)),'residual':float(np.linalg.norm(design(y,0,knots)[0]@co))})
 zones=[]
 for ci in [knots[0]/2,(knots[0]+knots[1])/2,(knots[1]+knots[2])/2,knots[2]+.05]:
  matrix=co[1:4,:].T.copy()
  for ki,knot in enumerate(knots):
   if ci>knot:matrix[:,0]+=co[7+ki]
  ev=np.linalg.eigvals(matrix);zones.append({'C_sample':float(ci),'eigenvalues':[[float(z.real),float(z.imag)] for z in ev],'max_real':float(max(ev.real))})
 out={'zone_eigenvalues':zones,'mode':can['mode'],'duration':a.duration,'detected_cycles':len(ok),'total':len(valid),'conditional_MAE':mae,'equilibria_at_zero':eigs,'records':valid,'scope':'Each gamma starts from reference cycle state, not a basin or unstable-orbit search; C, IP3 and h checked on solver nodes and sampled orbit; these checks do not prove invariance between samples; only untrained gamma values unless candidate has no training list.'};a.output.write_text(json.dumps(out,indent=2)+'\n')
 fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained');
 for j,keys in enumerate([('Cmin','Cmax'),('Imin','Imax'),('period',)]):
  for k in keys:
   ax[j].semilogx([r['gamma'] for r in valid],[r['reference'][k] for r in valid],'-o',ms=3,label='G-ChI '+k)
   ax[j].semilogx([r['gamma'] for r in ok],[r['candidate'][k] for r in ok],'x',ms=5,label='PWA '+k)
  ax[j].set_xlabel('Glu (µM)');ax[j].set_ylabel(['C (µM)','IP₃ (µM)','period (s)'][j]);ax[j].grid(alpha=.2);ax[j].legend(fontsize=7)
 fig.suptitle(f"{can['mode']}: cycles detected by forward integration ({len(ok)}/{len(valid)})");fig.savefig(a.output.with_suffix('.png'),dpi=170)
 print('SUMMARY',len(ok),len(valid),mae,flush=True)
if __name__=='__main__':main()
