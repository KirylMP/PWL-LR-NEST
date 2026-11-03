import sys,json,numpy as np
from scipy.linalg import null_space
from pathlib import Path
R=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R));from fit_modes import design
from poincare_fm_exact import evaluate_exact
p=json.load(open(str(R/'am_fold_joint/joint_refined.json')));s=json.load(open(str(R/'am_fold_joint/section25_seed.json')));k=p['knots'];g=s['gamma'];u=np.array(s['section_state']);co=np.array(p['coefficients']);N=null_space(design(np.array([x['state'] for x in p['fixed_equilibria']]),np.array([x['gamma'] for x in p['fixed_equilibria']]),k));w=(N.T@co).ravel();scale=np.array([.1,.5,1.]);cs=np.maximum(.3,np.minimum(abs(w),3.))
def f(v):
 q=evaluate_exact(N@v.reshape(7,3),k,g,u,c_section=.25,scan_step=.03);return np.r_[q['return_state']-u,q['fold_residual']]
J=np.empty((3,21));eps=5e-5
for j in range(21):
 v=np.eye(21)[j]*eps;J[:,j]=(f(w+v)-f(w-v))/(2*eps)
print('rank',np.linalg.matrix_rank(J),'norm',np.linalg.norm(f(w)),flush=True)
out={'gamma':g,'section_C':.25,'section_state':u.tolist(),'scaled_coefficient_jacobian':((J/scale[:,None])*cs[None,:]).tolist()};open(str(Path(__file__).resolve().parent/'section_analysis.json'),'w').write(json.dumps(out,indent=2))
