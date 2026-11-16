#!/usr/bin/env python3
"""Verify evidence reproducibility, media integrity and Pages-friendly packaging."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import xml.etree.ElementTree as ET

WEB=Path(__file__).resolve().parents[1];ROOT=WEB.parent
snapshot=json.loads((WEB/'public/data/evidence.json').read_text())
for key,source in snapshot['sources'].items():
    path=ROOT/source['path']
    assert hashlib.file_digest(open(path,'rb'),'sha256').hexdigest()==source['sha256'],f'Source drift: {key}; explicitly re-export evidence'
raw=json.loads((ROOT/snapshot['sources']['roundhouse']['path']).read_text())
assert raw['clips']==snapshot['roundhouse']['clips']
training=json.loads((ROOT/snapshot['sources']['training']['path']).read_text())
assert [r['loss_mean'] for r in training['history']]==[r['loss'] for r in snapshot['curve']]
assert len(snapshot['curve'])==21 and len(raw['clips'])==7
assert '/home/' not in (WEB/'public/data/evidence.json').read_text()
ET.parse(WEB/'public/method.svg')
config=json.loads((WEB/'scripts/highlights.json').read_text())
clip_count=len(config['clips']);montage_duration=sum(c['highlight'][1]-c['highlight'][0] for c in config['clips'])
extra=json.loads((WEB/'scripts/comparison_clips.json').read_text())
selections=config['clips']+extra['clips']
expected={'charleston','stance','climb','roundhouse','roundhouse-base','showreel'}
for clip in selections:expected.update([clip['id'],clip['id']+'-full'])
if (WEB/'public/media/scene-preview.mp4').exists():expected.add('scene-preview')
if (WEB/'public/media/policy-course-full.mp4').exists():expected.add('policy-course-full')
if (WEB/'public/media/native-stairs-full.mp4').exists():expected.add('native-stairs-full')
files=list((WEB/'public/media').glob('*.mp4'));assert {p.stem for p in files}==expected
pairs=json.loads((WEB/'public/data/comparisons.json').read_text())
assert set(pairs)=={c['id'] for c in config['clips']}
for pair in pairs.values():
    for mode in ('on','off'):
        meta=json.loads((WEB/'public/media'/(pair[mode+'Id']+'.json')).read_text())
        assert pair[mode+'Duration']==meta['duration_s']
        assert pair[mode+'SourceHash']==meta['source_sha256']
# Current moonwalk selection and the fresh no-reset aftermath recordings.
moon=json.loads((WEB/'public/data/moonwalk.json').read_text())
assert moon['stage']=='stage_500' and moon['actor_adam_updates_total']==12000
assert hashlib.sha256((ROOT/moon['source']).read_bytes()).hexdigest()==moon['source_sha256']
for skill in ('roll','vault','gaps','steps'):
    pair=pairs[skill]
    meta=json.loads((WEB/'public/media'/(pair['offId']+'.json')).read_text())
    v=meta['failure_visualization']
    assert v['resets']==v['teacher_queries']==v['training_updates']==0
    assert v['post_termination_seconds']==4 and v['evaluation_result_unchanged']
    assert v['native_termination_terms_preserved']
    assert abs(pair['offNativeEndS']-v['native_termination_s'])<1e-6
    assert meta['duration_s']>=pair['offNativeEndS']+3.95
    report=json.loads((ROOT/meta['report']).read_text())
    assert report['public_sha256']=='898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d'
for file in files:
    meta=json.loads(file.with_suffix('.json').read_text())
    assert file.with_suffix('.webp').stat().st_size>1000
    assert hashlib.file_digest(open(file,'rb'),'sha256').hexdigest()==meta['video_sha256']
    if meta.get('native_scene_highlights'):
        assert meta['edited_montage'] and meta['separate_recordings']
        assert not meta['continuous_episode'] and not meta['continuous_policy_rollout'] and not meta['single_rendered_scene']
        assert not meta['animated_transitions'] and not meta['spatial_retargeting'] and not meta['prop_visibility_edits']
        assert meta['camera_cuts']==3 and len(meta['chapters'])==4
        assert meta['racket_present_only_in_tennis'] and not meta['burned_in_labels']
        assert {c['id'] for c in meta['chapters']}=={'skate','steps','native-stairs','tennis'}
        assert len({c['source'] for c in meta['chapters']})==4
        assert len({c['scene'] for c in meta['chapters']})==4
        cursor=0
        for chapter in meta['chapters']:
            assert chapter['start_s']==cursor and chapter['native_scene_retained']
            assert chapter['racket_present']==(chapter['id']=='tennis')
            for field in ['source','scene','report']:
                assert hashlib.sha256((ROOT/chapter[field]).read_bytes()).hexdigest()==chapter[field+'_sha256']
            full_path=WEB/'public/media'/chapter['full_video'];assert full_path.exists()
            full=json.loads(full_path.with_suffix('.json').read_text())
            assert full['source_sha256']==chapter['source_sha256']
            assert chapter['excerpt_end_s']<=full['duration_s']+.001
            assert abs(chapter['end_s']-chapter['start_s']-(chapter['excerpt_end_s']-chapter['excerpt_start_s']))<1e-8
            cursor=chapter['end_s']
        assert cursor==meta['duration_s']==31
    elif meta.get('policy_course_highlights'):
        assert meta['edited_montage'] and meta['selected_excerpts']
        assert meta['continuous_episode'] is False and meta['continuous_policy_rollout'] is False
        assert meta['source_recording_continuous'] and not meta['animated_transitions']
        assert meta['camera_cuts']==3 and len(meta['chapters'])==4
        assert not meta['burned_in_labels']
        assert meta['racket_visible_only_in_tennis'] and meta['complete_source_preserves_original_prop_visibility']
        full_path=WEB/'public/media'/meta['full_video']
        assert hashlib.sha256(full_path.read_bytes()).hexdigest()==meta['full_video_sha256']
        full=json.loads(full_path.with_suffix('.json').read_text())
        assert full['physical_policy_course'] and full['source_sha256']==meta['source_sha256']
        source=ROOT/meta['source'];assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
        cursor=0
        for chapter in meta['chapters']:
            assert chapter['start_s']==cursor
            assert chapter['source_end_frame']<=full['output_frames']
            assert abs((chapter['source_end_frame']-chapter['source_start_frame'])/30-(chapter['end_s']-chapter['start_s']))<1e-8
            cursor=chapter['end_s']
        assert cursor==meta['duration_s']
    elif meta.get('physical_policy_course'):
        assert meta['policy_executed'] and meta['continuous_episode'] and meta['continuous_policy_rollout']
        assert meta['camera_cuts']==0 and not meta['animated_transitions']
        source=ROOT/meta['source'];assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
        report_path=ROOT/meta['report'];assert hashlib.sha256(report_path.read_bytes()).hexdigest()==meta['report_sha256']
        report=json.loads(report_path.read_text())
        assert report['success'] and report['actor_instances']==1
        assert report['robot_state_writes_after_start']==report['robot_resets_after_start']==report['teacher_action_queries']==report['training_updates']==0
        assert report['public_unchanged'] and report['public_sha256']==meta['public_sha256']
        assert len(report['tennis_cycles'])==3 and sum(c['racket_contact'] for c in report['tennis_cycles'])==2
        audit_path=ROOT/meta['physical_audit'];assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==meta['physical_audit_sha256']
        audit=json.loads(audit_path.read_text());assert audit['passed'] and audit['source_sha256']==meta['source_sha256']
        assert audit['robot_pose_writes_in_independent_replay_after_initialization']==0
        assert audit['max_root_qpos_error']==audit['max_joint_error_rad']==0
        for plugin in meta['plugins']:
            assert hashlib.sha256((ROOT/plugin['path']).read_bytes()).hexdigest()==plugin['sha256']
    elif meta.get('edited_montage'):
        assert meta['continuous_episode'] is False
        assert meta['burned_in_labels'] is False
        assert len(meta['chapters'])==clip_count and meta['duration_s']==montage_duration
        cursor=0
        for chapter in meta['chapters']:
            assert chapter['start_s']==cursor
            source=ROOT/chapter['source']
            assert hashlib.file_digest(open(source,'rb'),'sha256').hexdigest()==chapter['source_sha256']
            clip=json.loads((WEB/'public/media'/f"{chapter['id']}.json").read_text())
            assert chapter['excerpt_start_s']==clip['excerpt_start_s']
            assert chapter['excerpt_end_s']==clip['excerpt_end_s']
            cursor=chapter['end_s']
        assert cursor==montage_duration
    elif meta.get('design_previsualization'):
        assert meta['continuous_policy_rollout'] is False and meta['continuous_episode'] is False
        assert meta['single_rendered_scene'] and meta['camera_cuts']==0
        assert meta['animated_transitions'] and meta['spatially_retargeted_recorded_motions']
        assert len(meta['stages'])==7 and len(meta['animated_joins'])==6
        for source in meta['sources']:
            assert hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256']
        assert meta['pose_interpolation'] and meta['burned_in_labels'] is False
    else:
        source=ROOT/meta['source']
        assert hashlib.file_digest(open(source,'rb'),'sha256').hexdigest()==meta['source_sha256']
        if meta.get('full_video'):
            full=json.loads((WEB/'public/media'/meta['full_video']).with_suffix('.json').read_text())
            assert meta['source_sha256']==full['source_sha256']
            assert meta['partial_excerpt'] is True
            assert meta['excerpt_start_s']>=0 and meta['excerpt_end_s']<=full['duration_s']+.001
            selection=next(c for c in selections if c['id']==meta['id'])
            expected_duration=round((selection['highlight'][1]-selection['highlight'][0])*30)/30
            assert abs(meta['duration_s']-expected_duration)<1e-8
    result=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt','-of','json',str(file)]))['streams'][0]
    assert (result['codec_name'],result['width'],result['height'],result['r_frame_rate'],result['pix_fmt'])==('h264',1920,1080,'30/1','yuv420p')
    assert int(result['nb_frames'])==meta['output_frames']
    assert meta['speed']==1
    assert meta['pose_interpolation'] is bool(meta.get('design_previsualization',False))
    # Parse top-level ISO BMFF boxes; moov must precede media payload for faststart.
    boxes=[]
    with file.open('rb') as f:
        while True:
            header=f.read(8)
            if not header:break
            size,typ=struct.unpack('>I4s',header)
            if size==1:size=struct.unpack('>Q',f.read(8))[0];skip=size-16
            else:skip=size-8
            assert size>0
            boxes.append(typ);f.seek(skip,1)
    assert boxes.index(b'moov')<boxes.index(b'mdat')
    print(f'{file.name}: H.264 1080p/30, {meta["output_frames"]} frames, faststart, source/video hashes valid')
assert sum(f.stat().st_size for f in files)<150*1024*1024
print(f'PASS: {len(files)} videos including {clip_count} highlights, ON/OFF comparisons for every skill, complete source recordings/task segments and a {montage_duration}-second montage; source integrity, 21 exact training points, all 7 comparisons, valid SVG, no machine-local evidence paths.')
