#!/usr/bin/env python3
"""Convert a reduced fit result into a candidate consumable by existing diagnostics."""
import argparse,json,numpy as np
from fit_modes import design
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--template',type=Path,required=True);p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fold-output',type=Path);a=p.parse_args();d=json.loads(a.template.read_text());r=json.loads(a.fit.read_text());z=r['solution'];d['coefficients']=z['coefficients'];d['training_gammas']=sorted(set(d.get('training_gammas',[])+r['training_gammas']));d['all_zone_leading_eigenvalues']=z['leading_eig'];d['lower_zone_leading_eigenvalues']=z['leading_eig'][:2];states=np.array([q['state'] for q in d['fixed_equilibria']]);gammas=np.array([q['gamma'] for q in d['fixed_equilibria']]);d['max_anchor_residual']=float(np.max(abs(design(states,gammas,d['knots'])@np.array(z['coefficients']))));d.pop('fold_proxy',None);d.pop('fit_signature',None);d['provenance']='Local implicit first-return fold refinement; validate independently.';d['first_return_fold']={'gamma':z['gamma'],'section_C':z['section_C'],'section_state':z['section_state'],'return_time':z['return_time'],'equation_residual':z.get('fold_equation_residual'),'equation_residual_scaled':z.get('fold_equation_residual_scaled')};d['objective']=r['best_cost'];d['optimizer_success']=r['optimizer_success'];d['status']=r['status'];d['seed']=r['seed'];a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,indent=2)+'\n')
 if a.fold_output:
  a.fold_output.parent.mkdir(parents=True,exist_ok=True);a.fold_output.write_text(json.dumps(z,indent=2)+'\n')
if __name__=='__main__':main()
