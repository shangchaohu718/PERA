#!/usr/bin/env python3
"""Assemble real comparison sources and validate their initial states."""
import json
from pathlib import Path
import copy
import argparse
import numpy as np
from render_highlights import CONFIG, ROOT, WEB

def main():
    p=argparse.ArgumentParser();p.add_argument('--available-only',action='store_true');args=p.parse_args()
    clips=[];pairs={}
    for original in CONFIG['clips']:
        skill=original['id']
        if skill not in ('kick','moonwalk','tennis'):
            missing=[mode for mode in ('on','off') if not (ROOT/'website/.render-work/comparisons-v5'/(skill+'-'+mode)/'report.json').exists()]
            if missing:
                if args.available_only:continue
                raise FileNotFoundError((skill,missing))
        pairs[skill]={'label':original['name'],'onId':skill,'offId':skill+'-off',
                      'note':'Same task, seed and starting pose. Switching restarts the selected real recording.'}
        if skill=='moonwalk':pairs[skill]['note']='Same reference motion, seed and foot-friction protocol. Switching restarts the selected real recording.'
        if skill=='tennis':pairs[skill]['note']='Same seed and incoming-ball launch protocol. Switching restarts the selected real recording.'
        if skill=='kick':
            pairs[skill].update(offId='roundhouse-base',note='Same joint-motion reference and nominal physics. Refined expert versus the frozen public policy; camera origins are spatially aligned. The earlier quantitative comparison is shown separately below.')
            continue
        for enabled in ([False] if skill in ('moonwalk','tennis') else [True,False]):
            mode='on' if enabled else 'off';cid=skill+'-'+mode;c=copy.deepcopy(original)
            c.update(id=cid,family=skill,expert='Task expert ON' if enabled else 'Frozen public policy - Expert OFF',
                     name=original['name']+' - Expert '+mode.upper())
            if skill=='moonwalk':
                c['source']=original['source'].replace('/expert_eval/','/public_eval/')
                c['scope']='Frozen released public policy on the same repeated moonwalk reference and the same modified foot-friction protocol as the expert. Seed 42; actual closed-loop states.'
            else:
                folder=ROOT/'website/.render-work/comparisons-v5'/cid
                extended=ROOT/'website/.render-work/comparisons-v6'/cid
                if (extended/'report.json').exists():folder=extended
                report=json.loads((folder/'report.json').read_text())
                sub=folder/'recording' if skill in ('gaps','steps') and 'visualization' not in report else folder
                c.update(kind='native',source=str((sub/'physics.npz').relative_to(ROOT)),
                         scene=str((sub/'scene.mjb').relative_to(ROOT)),report=str((folder/'report.json').relative_to(ROOT)))
                with np.load(sub/'physics.npz') as f:duration=len(f['qpos'])/float(f['fps'])
                end=min(20.,duration) if skill=='tennis' else min(5.,duration) if skill=='skate' else duration if not enabled or skill in ('roll','vault') else min(6. if skill=='gaps' else 5.,duration)
                c.update(highlight=[0,end],poster_time=min(.6,end/2),recorded_times=skill=='tennis',
                         scope=f'Fresh frozen-policy standalone evaluation, expert {mode.upper()}; no training or teacher actions. Native failure/end preserved. Source report records actual duration and outcome.')
                if 'visualization' in report:
                    c['failure_visualization']=report['visualization']
                    if skill=='gaps':c['camera']['fall_elevation']=-65
                    if skill=='steps':c['camera'].update(fall_elevation=-65,fall_height=.3,fall_range=.5)
                    c['scope']='Native evaluation ends at %.2f s. Four further seconds show the same frozen public controller continuing without reset, solely to reveal the physical fall and aftermath; outside the scored evaluation.'%report['visualization']['native_termination_s']
                if skill=='tennis':c['hands_validation']=str((folder/'hands-validation.json').relative_to(ROOT))
            clips.append(c)
            if enabled:pairs[skill]['onId']=cid
        if skill not in ('moonwalk','tennis'):
            base=ROOT/'website/.render-work/comparisons-v5'
            suffix='recording/physics.npz' if skill in ('gaps','steps') else 'physics.npz'
            on=np.load(base/(skill+'-on')/suffix)['qpos'][0]
            off=np.load(base/(skill+'-off')/suffix)['qpos'][0]
            error=float(abs(on-off).max());assert error<1e-7,(skill,error)
            pairs[skill]['initial_qpos_max_difference']=error
    (WEB/'scripts/comparison_clips.json').write_text(json.dumps({'clips':clips,'pairs':pairs},indent=2)+'\n')
    print('Prepared',len(clips),'comparison exports for',len(pairs),'skills')

if __name__=='__main__':main()
