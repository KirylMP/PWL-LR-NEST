#!/usr/bin/env python3
"""3D G-ChI, De Pitta et al 2009 Eqs 5,6,20 and Table 1, AM."""
import numpy as np
from scipy.optimize import root, brentq
from scipy.integrate import solve_ivp

def field(t,y,gamma,vbeta=.2):
    C,h,I=y
    Q=1.049*(I+.13)/(I+.9434)
    m=I/(I+.13); n=C/(C+.08234)
    dc=(6*m**3*n**3*h**3+.11)*(2-1.185*C)-.9*C*C/(C*C+.1**2)
    dh=.2*(Q-(Q+C)*h)
    kgamma=1.3+10*C/(C+.6)
    di=vbeta*gamma**.7/(gamma**.7+kgamma**.7)+.02/(1+I/1.5)*C*C/(C*C+.1**2)-2*C**4/(C**4+.7**4)*I/(I+1)-.04*I
    return np.array([dc,dh,di])

def jac(y,g):
    return np.column_stack([np.imag(field(0,np.asarray(y,dtype=complex)+1e-25j*np.eye(3)[i],g))/1e-25 for i in range(3)])

def equilibrium(g,guess):
    r=root(lambda y:field(0,y,g),guess,jac=lambda y:jac(y,g),tol=1e-11)
    if not r.success and np.linalg.norm(r.fun)>1e-9: raise RuntimeError((g,r.message,r.x,r.fun))
    if np.min(r.x)<0: raise RuntimeError(('nonphysical',g,r.x))
    return r.x

def eigmax(y,g):return max(np.linalg.eigvals(jac(y,g)).real)

def solve_cycle(g,y,duration=900,rtol=1e-7,atol=1e-9):
    def peak(t,y,gamma):return field(t,y,gamma)[0]
    peak.direction=-1;peak.terminal=False
    s=solve_ivp(field,(0,duration),y,args=(g,),method='DOP853',rtol=rtol,atol=atol,max_step=.7,events=peak,dense_output=True)
    if not s.success:raise RuntimeError(s.message)
    t=s.t_events[0]; p=s.y_events[0]
    t=t[t>duration*.4];p=p[-len(t):] if len(t) else p[:0]
    data={'status':'not_converged','y_end':s.y[:,-1].tolist()}
    if len(t)<4:return data
    T=np.diff(t[-4:]); samples=s.sol(np.linspace(t[-2],t[-1],350)); amp=np.ptp(samples[0]);delta=np.linalg.norm(p[-1]-p[-2]);
    if np.max(np.abs(T-T[-1]))/T[-1]>.002 or delta>max(1e-5,.002*amp) or amp<1e-4:return data
    data.update(status='cycle',period=float(T[-1]),cmin=float(samples[0].min()),cmax=float(samples[0].max()),imin=float(samples[2].min()),imax=float(samples[2].max()),closure=float(delta),peak=p[-1].tolist())
    return data
