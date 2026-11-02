#!/usr/bin/env python3
"""Reproducible multistart refinement with trajectory and spectral penalties.

One case/seed per process; separate output directories permit independent jobs.
The penalties do not establish HLB1 or a fold: see README_SERVER.md.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from fit import KMAP,lr,eq,basis,upper,simulate,cycles

ROOT=Path(__file__).resolve().parent

def frozen_seed(d):
    a,b=d['a'],d['b']; n=np.asarray(d['alpha']); h=np.asarray(d['beta'])
    return np.array([[-a*n[0],b*h[0]],[-a*n[1],b*h[1]],[a,-b],[-a*n[2],b*h[2]],[-a*n[3],b*h[3]],[-a*n[4],b*h[4]]])

def reference_orbit(item,k,n):
    q=item['LR'];i=q['I'];T=q['period'];dur=max(300.,10*T)
    sol=simulate(lambda c,h,i:lr(c,h,i,k),i,q['initial'],dur)
    if not sol.success or sol.t[-1]<dur-1e-5:raise RuntimeError('LR reference integration failed')
    t=np.linspace(dur-3*T,dur,3001);c=sol.sol(t)[0];peaks,_=find_peaks(c,prominence=.003)
    if len(peaks)<2:raise RuntimeError(f'LR reference lacks peaks at I={i}')
    # Use one peak and one reported period. Include t=0 and t=T for closure.
    t0=t[peaks[-2]];ts=np.linspace(0,T,n)
    if t0+T>dur:raise RuntimeError('Insufficient reference trajectory')
    y=sol.sol(t0+ts)
    return {'I':i,'T':T,'initial':y[:,0].tolist(),'reference':y.T.tolist()}

def penalty(x,scale):return np.maximum(0.,x)/scale

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=list(KMAP),required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--start',choices=['frozen','field','mix'],default='mix')
    ap.add_argument('--max-nfev',type=int,default=300);ap.add_argument('--samples',type=int,default=28)
    ap.add_argument('--jitter',type=float,default=.07);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-seconds',type=float,default=0,help='cooperative time limit; 0 disables')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    k=KMAP[args.case];data=json.loads((ROOT/'data'/f'{args.case.lower()}_validation.json').read_text())
    base=json.loads((ROOT/'data'/f'{args.case.lower()}_parameters.json').read_text())
    fitted=json.loads((ROOT/'seeds'/f'{args.case.lower()}.json').read_text())
    train=[x for x in data['cycles'] if x['split']=='train'];orbits=[reference_orbit(x,k,args.samples) for x in train]
    eqi=np.linspace(max(0,min(x['I'] for x in data['cycles'])-.12),min(1.3,max(x['I'] for x in data['cycles'])+.12),21)
    eqs=np.array([eq(i,k) for i in eqi]); cuts=base['physical_thresholds_C']; ev=base['fixed_equilibrium_threshold_I']
    # Frozen initialization approximates the two horizontal thresholds, but the
    # first rises from C0 on [I1,I2]; thus it is not identical to the baseline.
    frozen_g=np.array([cuts[0],ev[0]-.10,.06,cuts[1]-cuts[0]])
    fg=fitted['geometry'];field_g=np.array([fg['C0'],fg['I1'],fg['I2']-fg['I1'],fg['height']])
    if args.start=='field' or (args.start=='mix' and args.seed%2):g=field_g.copy();co=np.array(fitted['coefficient_rows'])
    else:g=frozen_g.copy();co=frozen_seed(base)
    c0=cuts[0];l=np.array([max(.005,c0*.3),max(0,ev[0]-.15),.025,.015]);u=np.array([min(.5,c0*2.5),min(1.1,ev[0]+.15),.5,1.2])
    g=np.clip(g,l+1e-6,u-1e-6);rng=np.random.default_rng(args.seed)
    if args.seed:g=np.clip(g+rng.normal(0,args.jitter,size=4)*(u-l),l+1e-6,u-1e-6);co=co+rng.normal(size=co.shape)*args.jitter*np.maximum(np.abs(co),.15)
    x0=np.r_[g,co.ravel()];lo=np.r_[l,np.full(12,-80.)];hi=np.r_[u,np.full(12,80.)];x0=np.clip(x0,lo+1e-8,hi-1e-8)
    nper=args.samples*2+2
    started=time.monotonic();calls=0;best=np.inf;latest=None
    def evaluate(x):
        nonlocal calls,best,latest
        calls+=1;g=x[:4];co=x[4:].reshape(6,2)
        def f(c,h,i):return basis(c,h,i,g)@co
        parts=[]
        for o in orbits:
            i=o['I'];T=o['T'];ts=np.linspace(0,T,args.samples);ref=np.array(o['reference'])
            def rhs(t,z):return f(z[0],z[1],i)
            def stop(t,z):return min(z[0]+.05,2.1-z[0],z[1]+1.2,3.2-z[1])
            stop.terminal=True
            try:
                sol=solve_ivp(rhs,(0,T),o['initial'],t_eval=ts,rtol=2e-5,atol=2e-7,max_step=.35,events=stop)
                valid=sol.success and sol.y.shape[1]==len(ts) and np.isfinite(sol.y).all()
            except (ValueError,FloatingPointError):valid=False
            if valid:
                dy=(sol.y.T-ref)/np.array([.12,.12]);r=np.r_[dy.ravel()/np.sqrt(args.samples), (sol.y[:,-1]-o['initial'])/np.array([.12,.12])]
            else:r=np.full(nper,10.)
            parts.append(r/np.sqrt(len(orbits)))
        # Shape and equilibrium penalties are deliberately soft, not certificates.
        eqv=basis(eqs[:,0],eqs[:,1],eqi,g)@co
        parts.append((2*eqv/np.array([.12,.04])/np.sqrt(len(eqi))).ravel())
        # Jacobians share column h; require stable outer and unstable-focus middle.
        def matrix(c_slope):return np.column_stack((c_slope,co[2]))
        mats=[matrix(co[1]),matrix(co[1]+co[4]),matrix(co[1]+co[4]+co[5])]
        tl,tm,tu=[np.trace(a) for a in mats];dl,dm,du=[np.linalg.det(a) for a in mats]
        # enforce trace/determinant margins and negative discriminant in middle;
        # stable focus in upper is a desired hypothesis, not hard assertion.
        spec=np.r_[penalty(tl+.01,.1),penalty(.01-tm,.1),penalty(tu+.01,.1),penalty(.005-dl,.1),penalty(.005-dm,.1),penalty(.005-du,.1),penalty(tm*tm-4*dm+.005,.2),penalty(tu*tu-4*du+.005,.2)]
        parts.append(2*spec)
        # Keep coefficients near initialization to avoid unconstrained blow-up.
        parts.append(.015*(x[4:]-x0[4:]))
        residual=np.concatenate(parts);score=float(residual@residual)
        if score<best:
            best=score;latest=x.copy();tmp=args.out/'checkpoint.tmp';tmp.write_text(json.dumps({'case':args.case,'seed':args.seed,'status':'unverified_checkpoint','score':score,'calls':calls,'x':x.tolist()},indent=2));tmp.replace(args.out/'checkpoint.json')
        if calls==1 or calls%25==0:print(f'{args.case} seed={args.seed} calls={calls} loss={score:.5g} best={best:.5g} time={time.monotonic()-started:.1f}s',flush=True)
        if args.max_seconds and time.monotonic()-started>args.max_seconds:raise TimeoutError('time limit reached')
        return residual
    status='finished'
    try:
        fit=least_squares(evaluate,x0,bounds=(lo,hi),max_nfev=args.max_nfev,diff_step=1e-4,x_scale='jac',ftol=1e-6,xtol=1e-6,gtol=1e-6)
        optimizer={'success':bool(fit.success),'message':fit.message,'nfev':fit.nfev,'optimality':float(fit.optimality)}
    except TimeoutError:
        status='time_limit';optimizer={'success':False,'message':'cooperative time limit reached; using best checkpoint'}
    if latest is None:raise RuntimeError('No candidate evaluated')
    g=latest[:4];co=latest[4:].reshape(6,2)
    def f(c,h,i):return basis(c,h,i,g)@co
    comparison=[]
    for item in data['cycles']:
        a=item['LR'];dur=max(450,14*a['period']);sol=simulate(f,a['I'],a['initial'],dur)
        comparison.append({'I':a['I'],'split':item['split'],'target':{z:a[z] for z in ('period','Cmin','Cmax')},'candidate':cycles(sol,dur)})
    mats=[np.column_stack((co[1]+s,co[2])) for s in (0,co[4],co[4]+co[5])]
    eigen=[[{'real':float(y.real),'imag':float(y.imag)} for y in np.linalg.eigvals(m)] for m in mats]
    result={'status':status,'case':args.case,'K':k,'seed':args.seed,'start':args.start,'basis':'[1,C,h,I,max(C-C0,0),max(C-upper(I),0)]','geometry':{'C0':g[0],'I1':g[1],'I2':g[1]+g[2],'height':g[3]},'coefficient_rows':co.tolist(),'jacobians':[m.tolist() for m in mats],'eigenvalues':eigen,'training_objective':best,'optimizer':optimizer,'calls':calls,'validation':comparison,'scope':'soft spectral constraints and finite forward simulation; HLB1, fold and unstable branches unverified'}
    (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SAVED',args.out/'result.json',flush=True)
if __name__=='__main__':main()
