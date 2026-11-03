#!/usr/bin/env python3
"""Numerical nondegeneracy and transversality diagnostics of a Poincare fold candidate."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from poincare_fm_exact import evaluate_exact
from poincare_fm import evaluate,NoReturn

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--fold',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.candidate.read_text());fold=json.loads(a.fold.read_text());co=np.array(fold['coefficients']);kn=d['knots'];g=fold['gamma'];u=np.array(fold['section_state']);sec=fold['section_C'];
 def m(gamma,y,step=.025):return evaluate_exact(co,kn,gamma,y,c_section=sec,scan_step=step)
 base=m(g,u);DP=base['DP'];ev,V=np.linalg.eig(DP);j=int(np.argmin(abs(ev-1)));eig=float(ev[j].real);vr=V[:,j].real;vr/=np.linalg.norm(vr);le,W=np.linalg.eig(DP.T);w=W[:,int(np.argmin(abs(le-eig)))].real;w/=np.dot(w,vr);route=[x['threshold_index'] for x in base['switches']];est=[]
 for h in [2e-4,1e-4,5e-5]:
  a1=m(g,u+h*vr);a2=m(g,u-h*vr);gp=m(g+h,u);gm=m(g-h,u)
  same=all([x['threshold_index'] for x in z['switches']]==route for z in [a1,a2,gp,gm]);second=float(w@((a1['return_state']-2*base['return_state']+a2['return_state'])/h**2));trans=float(w@((gp['return_state']-gm['return_state'])/(2*h)));est.append({'step':h,'state_second_derivative':second,'parameter_derivative':trans,'route_unchanged':bool(same)})
 print('eigen',ev,'normal form',est,flush=True);last=est[1];alpha=last['parameter_derivative'];beta=last['state_second_derivative'];side=1 if -alpha/beta>0 else -1;branches=[]
 for dg in [side*.001,side*.01,side*.05]:
  gamma=g+dg;prediction=np.sqrt(max(0,-2*alpha*dg/beta))
  for sign in [-1,1]:
   start=u+sign*prediction*vr
   def fun(x):
    try:return m(gamma,x)['return_state']-x
    except NoReturn:return np.ones(2)*1e3
   sol=root(fun,start,tol=1e-9,options={'eps':1e-6,'maxfev':120});res=fun(sol.x)
   try:
    point=m(gamma,sol.x);rho=float(max(abs(np.linalg.eigvals(point['DP']))));row={'gamma':float(gamma),'initial_sign':sign,'section_state':sol.x.tolist(),'closure':float(np.linalg.norm(res)),'return_time':point['return_time'],'P_eigenvalues':[[float(z.real),float(z.imag)] for z in np.linalg.eigvals(point['DP'])],'spectral_radius':rho,'min_C_nodes':point['min_C_nodes'],'min_I_nodes':point['min_I_nodes'],'min_h_nodes':point['min_h_nodes'],'max_h_nodes':point['max_h_nodes'],'switches':point['switches']}
   except NoReturn as e:row={'gamma':float(gamma),'initial_sign':sign,'error':str(e)}
   branches.append(row);print('branch',gamma,sign,'closure',row.get('closure'),'rho',row.get('spectral_radius'),flush=True)
 out={'gamma':g,'section_state':u.tolist(),'P_eigenvalues':[[float(z.real),float(z.imag)] for z in ev],'right_eigenvector':vr.tolist(),'left_eigenvector':w.tolist(),'normal_form_estimates':est,'predicted_two_orbit_side':side,'branches':branches,'note':'Finite-difference numerical evidence only; step sensitivity, Poincare section, near-grazing, first-return branch and physical domain require independent checks. Not a formal proof.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
