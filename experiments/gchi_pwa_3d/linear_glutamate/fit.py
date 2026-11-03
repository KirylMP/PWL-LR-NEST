#!/usr/bin/env python3
"""Exploratory 3D PWA fits with one global linear glutamate term."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import least_squares
from scipy.signal import find_peaks

HERE=Path(__file__).resolve().parent
MODES=HERE.parent/'gchi_modes'
sys.path.insert(0,str(MODES))
from explore_modes import field,P

CAND={'AFM':MODES/'runs/afm-calcium-c1-01/candidate-2.json','FM':MODES/'runs/fm-topology-free-c1-01/candidate-3.json'}

def design(y,g,k):
 y=np.atleast_2d(y);g=np.broadcast_to(g,len(y))
 return np.column_stack((np.ones(len(y)),y,g,*[np.maximum(y[:,0]-q,0) for q in k]))

def matrix(co,z):
 A=co[1:4].T.copy()
 if z:A[:,0]+=np.sum(co[5:5+z],axis=0)
 return A

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['AFM','FM'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--max-nfev',type=int,default=80);ap.add_argument('--points',type=int,default=9);ap.add_argument('--duration',type=float,default=300);args=ap.parse_args()
 orig=json.loads(CAND[args.mode].read_text());k=np.array(orig['knots']);anchors=np.array([x['state'] for x in orig['fixed_equilibria']]);ag=np.array([x['gamma'] for x in orig['fixed_equilibria']]);D=design(anchors,ag,k);N=null_space(D)
 if np.linalg.matrix_rank(D)!=3 or N.shape!=(8,5):raise RuntimeError('Unexpected anchor rank')
 ref=json.loads((MODES/'results'/args.mode/'reference.json').read_text());cycles=[x for x in ref['cycles'] if x['status']=='cycle' and 'peak' in x];inds=np.unique(np.linspace(0,len(cycles)-1,min(len(cycles),args.points),dtype=int));xs=[];gs=[];ys=[];ds=[];starts=[];ends=[];seg_g=[];seg_t=[]
 for j in inds:
  c=cycles[j];g=c['gamma'];T=c['period'];t=np.linspace(0,T,33);sol=solve_ivp(field,(0,T),c['peak'],args=(g,P[args.mode]),t_eval=t,rtol=1e-9,atol=1e-11,method='DOP853',max_step=.2)
  if not sol.success:raise RuntimeError('reference solve failed')
  st=sol.y.T;xs.extend(st[:-1]);gs.extend([g]*32);ys.extend([field(0,x,g,P[args.mode]) for x in st[:-1]]);starts.extend(st[:-1]);ends.extend(st[1:]);seg_g.extend([g]*32);seg_t.extend([T/32]*32)
 X=np.array(xs);G=np.array(gs);Y=np.array(ys);start=np.array(starts);end=np.array(ends);seg_g=np.array(seg_g);dt=np.array(seg_t)/16;B=design(X,G,k)@N
 init=np.linalg.lstsq(B,Y,rcond=None)[0].ravel();rng=np.random.default_rng(args.seed)
 if args.seed:init=init+rng.normal(0,.002,len(init))
 fieldscale=np.array([.4,.05,.1]); states=np.array([.4,.15,.4]);counter=[0]
 def residual(v):
  counter[0]+=1;co=N@v.reshape(5,3);y=start.copy()
  def fun(w):return design(w,seg_g,k)@co
  with np.errstate(over='ignore',invalid='ignore'):
   for _ in range(16):
    a=fun(y);b=fun(y+dt[:,None]*a/2);c=fun(y+dt[:,None]*b/2);d=fun(y+dt[:,None]*c);y+=dt[:,None]*(a+2*b+2*c+d)/6
  if not np.all(np.isfinite(y)) or np.max(np.abs(y))>1e5:raise FloatingPointError
  ev=np.array([np.max(np.linalg.eigvals(matrix(co,j)).real) for j in range(4)])
  # stability signs stable/stable/unstable/stable; soft penalties
  spec=np.maximum(.005-np.array([-1.,-1.,1.,-1.])*ev,0.)/.1
  phys=np.column_stack((np.minimum(y[:,0],0.),np.minimum(y[:,2],0.),np.minimum(y[:,1],0.),np.maximum(y[:,1]-1,0.)))
  return np.r_[((B@v.reshape(5,3)-Y)/fieldscale).ravel()/np.sqrt(len(X)), 2*((y-end)/states).ravel()/np.sqrt(len(X)), 4*spec, 5*phys.ravel()/np.sqrt(len(X)),.0001*(v-init)]
 n=3*len(X)*2+4+4*len(X)+len(init)
 def safe(v):
  try:
   result=residual(v)
   if not np.all(np.isfinite(result)) or len(result)!=n:raise FloatingPointError
   return result
  except (FloatingPointError,OverflowError,ValueError,np.linalg.LinAlgError):return np.full(n,1e4)
 initial=safe(init)
 if len(initial)!=n or np.all(initial==1e4):raise RuntimeError('invalid initial guess')
 print('start',args.mode,'seed',args.seed,'points',len(inds),'residual norm',float(np.linalg.norm(initial)),flush=True)
 fit=least_squares(safe,init,max_nfev=args.max_nfev,x_scale='jac',ftol=1e-5,xtol=1e-5,verbose=1);co=N@fit.x.reshape(5,3)
 args.output.mkdir(parents=True,exist_ok=True)
 candidate={'mode':args.mode,'knots':k.tolist(),'coefficients':co.tolist(),'coefficient_basis':['1','C','h','IP3','gamma','hinge_C1','hinge_C2','hinge_C3'],'fixed_equilibria':orig['fixed_equilibria'],'anchor_residual':float(np.max(np.abs(D@co))),'training_gammas':[cycles[j]['gamma'] for j in inds],'spectral_max_real':[float(np.max(np.linalg.eigvals(matrix(co,j)).real)) for j in range(4)],'objective':float(np.sum(fit.fun**2)),'optimizer_success':bool(fit.success),'status':str(fit.message),'nfev':fit.nfev,'calls':counter[0], 'limitations':'One global linear gamma term; stability and physicality soft; fixed gamma only, not validated for dynamic glutamate; finite reference training segments, not a proven cycle or fold.'}
 (args.output/'candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
 records=[]
 for j in inds:
  q=cycles[j];g=q['gamma'];tt=np.linspace(0,args.duration,int(args.duration*10)+1)
  def rhs(t,y):return design(y,g,k)[0]@co
  def bad(t,y):return min(y[0]+1e-7,y[2]+1e-7,y[1]+1e-7,1+1e-7-y[1],10-np.linalg.norm(y))
  bad.terminal=True;bad.direction=-1
  sol=solve_ivp(rhs,(0,args.duration),q['peak'],t_eval=tt,events=bad,method='DOP853',rtol=1e-8,atol=1e-10,max_step=.25)
  tail=sol.y[:,sol.t>args.duration*.5];row={'gamma':g,'success':bool(sol.success),'t_end':float(sol.t[-1]),'physical_event':bool(len(sol.t_events[0])),'min_C':float(np.min(sol.y[0])),'min_I':float(np.min(sol.y[2]))}
  if tail.shape[1]>5:
   peaks,_=find_peaks(tail[0],prominence=.001);row.update({'Cmin':float(np.min(tail[0])),'Cmax':float(np.max(tail[0])),'n_peaks':int(len(peaks)),'period':float(np.median(np.diff(sol.t[sol.t>args.duration*.5][peaks]))) if len(peaks)>=2 else None})
  records.append(row);print('validation',row,flush=True)
 (args.output/'validation.json').write_text(json.dumps({'records':records,'duration':args.duration,'scope':'training gammas and reference-cycle initial states; finite integration, not independent validation or proof of cycle'},indent=2)+'\n')
 print('WROTE',args.output,flush=True)
if __name__=='__main__':main()
