#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
print('seed objective max_anchor_residual k fit_status validated cycles/25 below_H1 large-IC')
for directory in sorted([d for d in a.directory.iterdir() if d.is_dir()],key=lambda p:p.name):
 cand=directory/'candidate.json'; val=directory/'validation.json'
 if not cand.exists():print(directory.name,'NO_CANDIDATE (inspect checkpoint/log)');continue
 d=json.loads(cand.read_text());errs=[z['residual'] for z in d['fixed_equilibria']];v=json.loads(val.read_text()) if val.exists() else None
 print(d['seed'],round(d['best_loss'],6),f'{max(errs):.2e}',round(d['barrier']['k'],5),repr(d['optimizer_status']),'yes' if v else 'no',f"{v['detected_cycles']}/{v['total']}" if v else '-',v['below_H1']['trajectory']['status'] if v else '-')
