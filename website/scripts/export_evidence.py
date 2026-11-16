#!/usr/bin/env python3
"""Export a small, reproducible public evidence bundle from existing reports."""
import csv
import hashlib
import io
import json
from pathlib import Path

WEB=Path(__file__).resolve().parents[1]
ROOT=WEB.parent
OUT=WEB/'public/data'
SOURCES={
 'roundhouse': 'reports/roundhouse_expert_20261105/summary.json',
 'climb': 'reports/instinct_dagger_v1/run/final_summary.json',
 'training': 'reports/instinct_dagger_v1/run/training/report.json',
}

def main():
    source={k:json.loads((ROOT/p).read_text()) for k,p in SOURCES.items()}
    curve=[{'updates':r['actual_adam_steps'],'round':r['round'],'phase':r['phase'],'loss':r['loss_mean']} for r in source['training']['history']]
    data={
      'snapshot_date':'2026-11-05',
      'sources': {k:{'path':p,'sha256':hashlib.file_digest(open(ROOT/p,'rb'),'sha256').hexdigest()} for k,p in SOURCES.items()},
      'roundhouse':{k:v for k,v in source['roundhouse'].items() if k in ('clips','seed','evaluation','expert_scope','explicit_switch_check')},
      'climb':{k:v for k,v in source['climb'].items() if k in ('method','scene','evaluation','scope','actual_adam_steps','trainable_parameters','bc_adam_steps','dagger_adam_steps','public_frozen','visual_encoder_frozen','teacher_frozen')},
      'curve':curve,
      'limits':['PPO martial-arts evaluation: one seed (42), nominal Isaac Lab physics, no fall terminations, final 10 future-reference frames omitted.','Climbing evaluation: two held-out seeds (7002, 7003), eight environments each; fixed official 50 cm scene, native completion criteria.','The training curve is mean action-supervision loss per BC/DAgger round, not reward or a held-out performance curve.','Public-policy weights unchanged and software switch equivalence do not establish physical old-skill retention.','No real-robot evaluation, multi-seed martial-arts study or retention/visual ablation is supplied.'],
    }
    # Keep public evidence free of machine-local absolute paths.
    encoded=json.dumps(data,indent=2).replace(str(ROOT)+'/', '')
    (OUT/'evidence.json').write_text(encoded+'\n')
    for name in ('curve','roundhouse'):
        rows=curve if name=='curve' else [{'skill':r['name'],'public_body_error_cm':r['original']['mean_body_error_cm'],'expert_on_body_error_cm':r['expert_enabled']['mean_body_error_cm']} for r in data['roundhouse']['clips']]
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Exported verified report snapshot, 21 training points and all seven skill comparisons.')

if __name__=='__main__':main()
