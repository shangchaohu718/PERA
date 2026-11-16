#!/usr/bin/env python3
"""Render assembled real ON/OFF states and export the website comparison index."""
import json
from render_highlights import WEB, ROOT, MEDIA, WORK, QA, render

WORK.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
config=json.loads((WEB/'scripts/comparison_clips.json').read_text())
for clip in config['clips']:
    if (MEDIA/(clip['id']+'.json')).exists():continue
    render(clip)
pairs=config['pairs']
for skill,pair in pairs.items():
    for mode in ('on','off'):
        meta=json.loads((MEDIA/(pair[mode+'Id']+'.json')).read_text())
        pair[mode+'Duration']=meta['duration_s']
        pair[mode+'FullId']=meta.get('full_video',pair[mode+'Id']+'.mp4').removesuffix('.mp4')
        pair[mode+'Note']=meta.get('scope','Complete recorded evaluation, ending at its actual native termination.')
        pair[mode+'Source']=meta['source'];pair[mode+'SourceHash']=meta['source_sha256']
        if skill in ('roll','vault','gaps','steps','skate'):
            report=json.loads((ROOT/meta['report']).read_text())
            if skill in ('gaps','steps'):
                episode=report['episodes'][0];success=episode['success'];end=episode['elapsed_s']
            else:
                success=report['successful_episodes']>0;end=report['episode_seconds'][0]
            if skill=='skate':
                pair[mode+'Outcome']=f'Full {end:.0f} s evaluation: board travelled {report["board_displacement_m"][0]:.2f} m'
            else:
                timed_out=skill in ('gaps','steps') and episode['fired']['time_out']
                pair[mode+'Outcome']='Native task completed' if success else f'Native horizon reached at {end:.2f} s; task target not completed' if timed_out else f'Native evaluation terminated at {end:.2f} s'
            if 'visualization' in report:
                pair[mode+'NativeEndS']=report['visualization']['native_termination_s']
                pair[mode+'ContinuationS']=report['visualization']['post_termination_seconds']
                pair[mode+'Outcome']+= '; continued without reset'
        elif skill=='moonwalk':
            report=json.loads((ROOT/'reports/moonwalk_expert_20261105/stage_500/summary.json').read_text())
            value=report['expert' if mode=='on' else 'public']['mean_body_error_cm']
            pair[mode+'Outcome']=f'Full-run mean body error: {value:.2f} cm'
        else:pair[mode+'Outcome']='Continuous incoming-ball practice' if skill=='tennis' else 'Actual recorded motion replay'
output=MEDIA.parent/'comparisons.json'
output.write_text(json.dumps(pairs,indent=2)+'\n')
print('Comparison index ready:',len(pairs),'skills')
