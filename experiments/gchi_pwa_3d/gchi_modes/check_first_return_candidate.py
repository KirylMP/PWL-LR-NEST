#!/usr/bin/env python3
"""Independent adaptive check of first-return closure, fold condition and stability."""
import argparse,json
from pathlib import Path
import numpy as np
from poincare_fm import evaluate,NoReturn
from poincare_fm_exact import evaluate_exact
from fit_modes import design

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();d=json.loads(a.candidate.read_text());co=np.array(d['coefficients']);kn=np.array(d['knots']);q=d['first_return_fold'];g=q['gamma'];u=np.array(q['section_state']);sec=q['section_C'];anchors=np.array([r['state'] for r in d['fixed_equilibria']]);ag=np.array([r['gamma'] for r in d['fixed_equilibria']]);out={'gamma':g,'section_C':sec,'section_state':u.tolist(),'anchor_residual':float(np.max(abs(design(anchors,ag,kn)@co)))}
 try:
  z=evaluate(co,kn,g,u,c_section=sec,max_step=.025,rtol=2e-10,atol=2e-12);e=evaluate_exact(co,kn,g,u,c_section=sec,scan_step=.025);ev=np.linalg.eigvals(z['DP']);out.update(closure=float(np.linalg.norm(z['return_state']-u)),fold_determinant=z['fold_residual'],return_time=z['return_time'],return_map_eigenvalues=[[float(x.real),float(x.imag)] for x in ev],exact_vs_adaptive_DP=float(np.max(abs(z['DP']-e['DP']))),exact_vs_adaptive_return=float(np.max(abs(z['return_state']-e['return_state']))),min_C_nodes=z['min_C_nodes'],min_I_nodes=z['min_I_nodes'],min_h_nodes=z['min_h_nodes'],max_h_nodes=z['max_h_nodes'],switches=z['switches']);out['physical_at_checked_nodes']=bool(z['min_C_nodes']>=-1e-8 and z['min_I_nodes']>=-1e-8 and z['min_h_nodes']>=-1e-8 and z['max_h_nodes']<=1+1e-8)
 except NoReturn as ex:out['error']=str(ex)
 A=co[1:4].T.copy();leads=[]
 for j in range(4):
  if j:A[:,0]+=co[6+j]
  leads.append(float(max(np.linalg.eigvals(A).real)))
 out['region_max_real_eig']=leads;out['scope']='Finite sampled numerical checks; no global positivity, nondegeneracy, transversality or exhaustive orbit-branch proof.';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
