#!/usr/bin/env python3
"""Fixed anchors, calcium barrier, amplitude/period and an actual-orbit fold residual."""
import json,time,argparse,sys,os
import scipy
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import null_space
from fit_horizontal_basis import basis
from fit import generate
from reference_gchi import equilibrium
P=Path(__file__).parent
ap=argparse.ArgumentParser(description='One independent G-ChI multistart job');ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--max-nfev',type=int,default=120);ap.add_argument('--jitter',type=float,default=.02);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
seed=json.loads((P/'seed_guard_opt.json').read_text());co0=np.array(seed['coefficients']);geom=np.array(list(seed['geometry'].values()));hs=json.loads((P.parent/'gchi_am/output/hopf.json').read_text());fix=np.array([equilibrium(0,[.07,.8,.15]),*[np.array([h[k] for k in ('C','h','I')]) for h in hs]]);G=np.array([0,*[h['gamma'] for h in hs]]);D=basis(fix,G,geom);N=null_space(D);Dpinv=np.linalg.pinv(D);
def anchor_adjustment(threshold):
 Q=np.zeros((9,3));Q[:,0]=-Dpinv@np.maximum(threshold-fix[:,0],0);return Q
X,g,Y,w=generate();scales=np.array([.12,.025,.025]);targets=json.loads((P/'actual_rest_targets_extended.json').read_text());gs0=np.array([r['gamma'] for r in targets]);T0=np.array([r['period'] for r in targets]);yref=np.array([r['samples'] for r in targets]); yinit0=yref[:,0,:];stable_idx=[i for i,r in enumerate(targets) if r['kind']=='stable'];fi=[r['kind'] for r in targets].index('fold');gfold=gs0[fi];ref_fold=yinit0[fi];tfold=T0[fi];gs=np.r_[gs0,gfold];steps=20;cb=.1;k=2.;sc=np.array([.12,.12,.15]); start=time.monotonic();calls=0;best=np.inf

def ff(y,co):
 y=np.clip(y,-2,3);d=basis(y,gs,geom)@co;d[:,0]+=k*np.maximum(cb-y[:,0],0);return np.clip(np.nan_to_num(d,nan=1e3,posinf=1e3,neginf=-1e3),-100,100)
def jac_y(y,co):
 C,h,I=y;c0,i1,width,height=geom; slope=0.;top=c0+height;A=np.column_stack((co[1],co[2],co[3])).copy()
 if C<cb:A[0,0]-=k
 if C>c0:A[:,0]+=co[7]
 if C>top:A[:,0]+=co[8];A[:,2]-=co[8]*slope
 return A

def rhs(y,phi,co):return ff(y,co),jac_y(y[-1],co)@phi

def orbit(co,yfold,Tfold):
 y=np.vstack([yinit0,yfold]);T=np.r_[T0,Tfold];dt=T/(12*steps);ph=np.eye(3);tra=[y.copy()];fm=None
 for j in range(24):
  for _ in range(steps):
   a,ap=rhs(y,ph,co);b,bp=rhs(y+dt[:,None]*a/2,ph+dt[-1]*ap/2,co);c,cp=rhs(y+dt[:,None]*b/2,ph+dt[-1]*bp/2,co);d,dp=rhs(y+dt[:,None]*c,ph+dt[-1]*cp,co);y+=dt[:,None]*(a+2*b+2*c+d)/6;ph+=dt[-1]*(ap+2*bp+2*cp+dp)/6
  tra.append(y.copy())
  if j==11:fm=ph.copy()
 return np.stack(tra,axis=1),fm

def sjac(co,z):
 if z==0:return jac_y([(cb+geom[0])/2,.7,.36],co)
 if z==1:return jac_y([.2,.7,.36],co)
 if z==2:return jac_y([.25,.7,.8],co)
 return jac_y([.5,.7,.8],co)
def spec(A,sign):
 e=np.linalg.eigvals(A);lead=max(e.real);pair=e[np.argmax(abs(e.imag))];return [max(0,.012-sign*lead)/.08,max(0,.2-abs(pair.imag))/.4]

probe_gammas=np.repeat(np.array([.0115,.0123,.01275,.0129]),2);probe_ini=np.tile(np.array([[.22,.7,.36],[.32,.65,.45]]),(4,1));probe_ref=np.array([equilibrium(float(z),[.15,.72,.34]) for z in probe_gammas]);
def probe_field(y,co):
 d=basis(y,probe_gammas,geom)@co;d[:,0]+=k*np.maximum(cb-y[:,0],0);return np.clip(np.nan_to_num(d,nan=100.,posinf=100.,neginf=-100.),-100.,100.)
def probe_trajectory(co):
 y=probe_ini.copy();dt=.15;out=[];min_c=y[:,0].copy()
 for step in range(800):
  a=probe_field(y,co);b=probe_field(y+dt*a/2,co);c=probe_field(y+dt*b/2,co);d=probe_field(y+dt*c,co);y=y+dt*(a+2*b+2*c+d)/6
  min_c=np.minimum(min_c,y[:,0])
  if step in (532,665,799):out.append(y.copy())
 return np.array(out),min_c

def loss(v):
 global calls,best,k,cb
 calls+=1;k=float(v[22]);cb=float(v[23]);co=N@v[:18].reshape(6,3)+k*anchor_adjustment(cb);yf=v[18:21];Tfold=v[21]
 traj,M=orbit(co,yf,Tfold);diff=(traj[:-1,:13,:]-yref)/sc;closed=(traj[:-1,12,:]-yinit0)/sc;long=(traj[stable_idx,24,:]-yinit0[stable_idx])/sc
 ftraj=traj[-1,:13,:]; fc=(ftraj[-1]-yf)/sc; phase=ff(np.vstack([yinit0,yf]),co)[-1,0]/.1;tr=np.trace(M);p2=(tr**2-np.trace(M@M))/2;second=3-2*tr+p2
 pred=basis(X,g,geom)@co;pred[:,0]+=k*np.maximum(cb-X[:,0],0);field=(((pred-Y)*w[:,None])/scales).ravel();loweig=max(np.linalg.eigvals(jac_y(fix[0],co)).real);lowpen=max(0,loweig+.008)/.08;spectra=np.r_[spec(sjac(co,0),-1),spec(sjac(co,1),1),spec(sjac(co,2),1),spec(sjac(co,3),-1)]
 probe_values,probe_min=probe_trajectory(co);probe=(probe_values-probe_ref)/np.array([.1,.1,.1]);negative_penalty=np.maximum(-probe_min,0)/.05;amin=float(ftraj[:,0].min()); cminpen=max(0,.162-amin)/.05
 z=np.r_[.5*field,1.2*spectra,2*lowpen,(1.5*probe/np.sqrt(probe.size)).ravel(),(3*negative_penalty/np.sqrt(len(negative_penalty))),(2*diff/np.sqrt(diff.size)).ravel(),(1.5*closed/np.sqrt(closed.size)).ravel(),(2*long/np.sqrt(long.size)).ravel(),3*fc,2*phase,1.7*second,1.2*((yf-ref_fold)/sc),1.2*(Tfold-tfold),2*cminpen,.01*(co-co0).ravel()];z=np.clip(np.nan_to_num(z,nan=1e4,posinf=1e4,neginf=-1e4),-1e4,1e4)
 val=float(z@z)
 if val<best:
  best=val;temp=args.out/'checkpoint.tmp.npz';np.savez(temp,v=v,cost=val,calls=calls);temp.replace(args.out/'checkpoint.npz')
 if calls%50==1:print('calls',calls,'loss',round(val,5),'best',round(best,5),'time',round(time.monotonic()-start,1),'closure',round(float(np.linalg.norm(fc)),5),'foldres',round(float(second),5),'cmin',round(amin,5),flush=True)
 return z
k0=float(seed['barrier']['k']);cb0=float(seed['barrier']['Cb']);v0=np.r_[(N.T@(co0-k0*anchor_adjustment(cb0))).ravel(),ref_fold,tfold,k0,cb0];lower=np.r_[np.full(18,-30),[.35,.4,.4],6.,.1,.075];upper=np.r_[np.full(18,30),[.7,.85,1.1],12.,10.,.145]
rng=np.random.default_rng(args.seed)
if args.seed:
 v0[:18]+=rng.normal(0,args.jitter,18);v0[18:21]+=rng.normal(0,args.jitter*.05,3);v0[21]+=rng.normal(0,args.jitter*.2);v0[22]+=rng.normal(0,args.jitter*.3);v0[23]+=rng.normal(0,args.jitter*.5)
v0=np.clip(v0,lower+1e-7,upper-1e-7)
sol=least_squares(loss,v0,bounds=(lower,upper),max_nfev=args.max_nfev,ftol=2e-5,xtol=2e-5,gtol=2e-5,diff_step=3e-5,x_scale='jac');ck=np.load(args.out/'checkpoint.npz');v=ck['v'];k=float(v[22]);cb=float(v[23]);co=N@v[:18].reshape(6,3)+k*anchor_adjustment(cb);out={'geometry':seed['geometry'],'coefficients':co.tolist(),'fixed_equilibria':[{'gamma':float(gg),'state':y.tolist(),'residual':float(np.linalg.norm((basis(y,gg,geom)@co)[0]+np.array([k*max(cb-y[0],0),0,0])))} for gg,y in zip(G,fix)],'barrier':{'Cb':cb,'k':k},'fold_seed':{'gamma':float(gfold),'state':v[18:21].tolist(),'period':float(v[21])},'best_loss':float(ck['cost']),'calls':calls,'optimizer_status':sol.message,'threshold_type':'horizontal C0 and C0+height; I1 and width unused','starting_model':'seed_guard_opt.json','threshold_search_interval':[.075,.145],'near_h1_probe':{'gammas':[.0115,.0123,.01275,.0129],'initials':[[.22,.7,.36],[.32,.65,.45]],'times_s':[79.95,99.9,120.0],'reference':'G-ChI equilibrium; soft penalties for residual and negative C along the probe'},'seed':args.seed,'jitter':args.jitter,'max_nfev':args.max_nfev,'numpy_version':np.__version__,'scipy_version':scipy.__version__,'python_version':sys.version,'limitations':'RK4 optimization proxy; only reference trajectory data, no validated PWA fold. Best point may be a finite-difference probe; validate independently.'};(args.out/'candidate.json').write_text(json.dumps(out,indent=2)+'\n');print('FINISHED',sol.message,'calls',calls,'best',float(ck['cost']),flush=True)
