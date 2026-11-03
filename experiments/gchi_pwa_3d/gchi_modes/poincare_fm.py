#!/usr/bin/env python3
"""First positively directed Poincare return for a 3-D continuous PWA field.

Fixed glutamate; section C=C_section away from switching planes. On transverse
crossings the PWA field is continuous, so switching saltation is the identity.
Numerical solver handles the jump of the variational Jacobian adaptively.
"""
import numpy as np
from scipy.integrate import solve_ivp
from fit_modes import design

class NoReturn(RuntimeError):pass

def evaluate(co,knots,gamma,u,c_section=.2,tmax=200.,rtol=2e-9,atol=2e-11,max_step=.05):
 co=np.asarray(co,float);knots=np.asarray(knots,float);u=np.asarray(u,float);y0=np.array([c_section,*u])
 if min(abs(c_section-knots))<1e-7:raise NoReturn('section intersects a switching plane')
 def f(y):return design(y,gamma,knots)[0]@co
 if f(y0)[0] <= 1e-6:raise NoReturn('section is not crossed positively and transversely')
 def rhs(t,w):
  y=w[:3];A=co[1:4].T.copy()
  for j,k in enumerate(knots):
   if y[0]>k:A[:,0]+=co[7+j]
  return np.r_[f(y),(A@w[3:].reshape(3,3)).ravel()]
 def section(t,w):return w[0]-c_section
 section.direction=1;section.terminal=True
 def guard(t,w):return min(w[0]+.05,3-w[0],w[1]+.2,2-w[1],w[2]+.05,5-w[2])
 guard.terminal=True
 def threshold_event(k):
  def event(t,w):return w[0]-k
  event.direction=0;event.terminal=False;return event
 threshold_events=[threshold_event(k) for k in knots]
 # A short warmup prevents the initial point being counted as its own return.
 initial=np.r_[y0,np.eye(3).ravel()];eps=1e-6
 first=solve_ivp(rhs,(0,eps),initial,rtol=rtol,atol=atol,method='DOP853',max_step=eps)
 if not first.success:raise NoReturn('warmup failed')
 s=solve_ivp(rhs,(eps,tmax),first.y[:,-1],rtol=rtol,atol=atol,method='DOP853',max_step=max_step,events=[section,guard,*threshold_events])
 if not s.success or len(s.t_events[0])!=1:raise NoReturn('first return unavailable or domain guard hit')
 yf=s.y_events[0][0,:3];M=s.y_events[0][0,3:].reshape(3,3);fend=f(yf);speed=float(fend[0])
 if speed<=1e-6:raise NoReturn('return is not transverse')
 switches=[]
 for j in range(len(knots)):
  for tt,w in zip(s.t_events[2+j],s.y_events[2+j]):
   crossing_speed=float(f(w[:3])[0]);switches.append({'threshold_index':j,'time':float(tt),'speed':crossing_speed})
 if switches and min(abs(x['speed']) for x in switches)<1e-5:raise NoReturn('near grazing at a switching plane')
 switches.sort(key=lambda x:x['time'])
 # Projection corrects for the return-time variation; section coordinates are (h,I).
 projection=np.eye(3)-np.outer(fend,np.array([1.,0.,0.]))/speed
 DP=(projection@M)[1:,:][:,1:]
 return {'initial':y0,'end':yf,'return_time':float(s.t_events[0][0]),'return_state':yf[1:],'DP':DP,'fold_residual':float(np.linalg.det(DP-np.eye(2))),'crossing_speed':speed,'switches':switches,'min_C_nodes':float(min(first.y[0].min(),s.y[0].min())),'min_I_nodes':float(min(first.y[2].min(),s.y[2].min())),'min_h_nodes':float(min(first.y[1].min(),s.y[1].min())),'max_h_nodes':float(max(first.y[1].max(),s.y[1].max()))}

def section_point_from_peak(co,knots,gamma,peak,c_section=.2,tmax=200):
 def f(t,y):return design(y,gamma,knots)[0]@co
 def event(t,y):return y[0]-c_section
 event.direction=1;event.terminal=True
 s=solve_ivp(f,(0,tmax),peak,rtol=2e-10,atol=2e-12,method='DOP853',max_step=.04,events=event)
 if not s.success or not len(s.t_events[0]):raise NoReturn('could not find section from given peak')
 return s.y_events[0][0,1:]
