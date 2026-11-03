#!/usr/bin/env python3
"""First-return map using affine matrix exponentials and scalar crossing solves.

A scan brackets switching and section crossings; each bracket is refined with
Brent's method. Near-grazing and multiple crossings inside a scan step can be
missed; repeat with smaller scan steps and validate against an adaptive solver.
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from fit_modes import design
from poincare_fm import NoReturn

def evaluate_exact(co,knots,gamma,u,c_section=.2,tmax=200.,scan_step=.05):
 co=np.asarray(co,float);knots=np.asarray(knots,float);y=np.array([c_section,*u],float);M=np.eye(3);t=0.;sections=[];switches=[];zone_times=np.zeros(4);mins=y.copy();maxs=y.copy()
 if min(abs(c_section-knots))<1e-7:raise NoReturn('section on switching plane')
 def f(y):return design(y,gamma,knots)[0]@co
 if f(y)[0]<=1e-6:raise NoReturn('initial section crossing not positive transverse')
 matrices=[];aug=[];step=[];linear=[]
 cs=[knots[0]/2,(knots[0]+knots[1])/2,(knots[1]+knots[2])/2,knots[2]+.1]
 for j,c in enumerate(cs):
  A=co[1:4].T.copy()
  for k in range(j):A[:,0]+=co[7+k]
  ref=np.array([c,.7,1.]);b=f(ref)-A@ref;B=np.zeros((4,4));B[:3,:3]=A;B[:3,3]=b;matrices.append(A);aug.append(B);step.append(expm(B*scan_step));linear.append(expm(A*scan_step))
 def region(x,vel=None):
  if vel is not None and min(abs(x-knots))<1e-8:return int(np.searchsorted(knots,x+np.sign(vel)*1e-8,side='right'))
  return int(np.searchsorted(knots,x,side='right'))
 def propagate(j,x,delta):return (expm(aug[j]*delta)@np.r_[x,1])[:3]
 def advance_small(j,x,delta):return propagate(j,x,delta),expm(matrices[j]*delta)
 j=region(y[0]);h=1e-7;y,E=advance_small(j,y,h);M=E@M;t+=h;zone_times[j]+=h
 while t<tmax:
  h=min(scan_step,tmax-t);B=step[j] if abs(h-scan_step)<1e-14 else expm(aug[j]*h);F=linear[j] if abs(h-scan_step)<1e-14 else expm(matrices[j]*h);yn=(B@np.r_[y,1])[:3]
  events=[]
  for k in [*knots,c_section]:
   f0=y[0]-k;f1=yn[0]-k
   if f0*f1<0:
    tau=brentq(lambda q:propagate(j,y,q)[0]-k,0,h,xtol=2e-13);ye=propagate(j,y,tau);speed=f(ye)[0]
    if abs(speed)<1e-5:raise NoReturn('near grazing in crossing bracket')
    if k!=c_section or speed>0:events.append((tau,k,ye,speed))
  if not events:
   y=yn;M=F@M;t+=h;zone_times[j]+=h;mins=np.minimum(mins,y);maxs=np.maximum(maxs,y)
   if y[0]<-.05 or y[0]>3 or y[2]<-.05 or y[2]>5 or y[1]<-.2 or y[1]>2:raise NoReturn('domain guard hit')
   continue
  tau,k,ynew,speed=min(events,key=lambda z:z[0]);M=expm(matrices[j]*tau)@M;t+=tau;zone_times[j]+=tau;y=ynew;mins=np.minimum(mins,y);maxs=np.maximum(maxs,y)
  if k==c_section:
   proj=np.eye(3)-np.outer(f(y),np.array([1.,0.,0.]))/speed;DP=(proj@M)[1:,1:]
   return {'initial':np.array([c_section,*u]),'end':y,'return_time':float(t),'return_state':y[1:],'DP':DP,'fold_residual':float(np.linalg.det(DP-np.eye(2))),'crossing_speed':float(speed),'switches':switches,'zone_times':zone_times,'min_C_nodes':float(mins[0]),'max_C_nodes':float(maxs[0]),'min_I_nodes':float(mins[2]),'max_I_nodes':float(maxs[2]),'min_h_nodes':float(mins[1]),'max_h_nodes':float(maxs[1]),'scan_step':float(scan_step)}
  idx=int(np.argmin(abs(knots-k)));switches.append({'threshold_index':idx,'time':float(t),'speed':float(speed)});j=region(y[0],speed);h=min(1e-7,tmax-t);y,E=advance_small(j,y,h);M=E@M;t+=h;zone_times[j]+=h;mins=np.minimum(mins,y);maxs=np.maximum(maxs,y)
 raise NoReturn('no first return before tmax')
