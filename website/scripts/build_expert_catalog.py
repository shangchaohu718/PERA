"""Export the 31-task gallery from verified legacy records, never invented replays.

Provenance stays in downloadable data/sidecars; presentation uses neutral skill
names and ON/OFF labels. Run via scripts/run_flex_supermimic.sh for native replay.
"""
from pathlib import Path
import argparse
import json
import hashlib
import subprocess
import math
import os

WEB=Path(__file__).resolve().parents[1]
ROOT=WEB.parent
WORK=WEB/'.render-work/catalog31-v1'
MEDIA=WEB/'public/media/catalog31'
SOURCE_TEXT_EN=json.loads((WEB/'scripts/catalog_english.json').read_text())
BASE='898d1a02829f901fbad4ed07178df2b06778620b54e8bccf5e07ffcb4c31d32d'
GROUPS=[{'id':'body','label':'Rolls & climbs'},{'id':'terrain','label':'Locomotion & terrain'},
        {'id':'sports','label':'Skating & ball sports'},{'id':'objects','label':'Object handling'},
        {'id':'motions','label':'Martial arts'}]
EXCLUDED={'diveroll_lab_wall','diveroll_ramp_wall','knee_climb_wall',
          'jump_sit_wall','roll_vault_wall',
          'mimicx_descent','mimicx_bench','mimicx_forest','mimicx_football',
          'mimicx_tennis','mimicx_dance','mimicx_kungfu'}
LEGACY={'diveroll_lab_no_wall':'roll','roll_vault_no_wall':'vault',
        'husky_skateboard':'skate','parkour_stairs_up_high':'steps',
        'parkour_gaps':'gaps','athlete_tennis':'tennis','roundhouse':'kick'}
LABELS={
 'diveroll_lab_no_wall':('Forward roll','No wall · Laboratory'),
 'diveroll_ramp_no_wall':('Ramp roll','No wall · Ramp'),
 'diveroll_lab_wall':('Forward roll with wall','Wall · Laboratory'),
 'diveroll_ramp_wall':('Ramp roll with wall','Wall · Ramp'),
 'knee_climb_no_wall':('Step climb','No wall'),
 'knee_climb_wall':('Step climb with wall','Wall'),
 'jump_sit_no_wall':('Jump to sit','No wall'),
 'jump_sit_wall':('Jump to sit with wall','Wall'),
 'roll_vault_no_wall':('Obstacle vault','No wall'),
 'roll_vault_wall':('Vault with wall','Wall'),
 'velocity_walk':('Commanded walking','Walk · Turn · Stop'),
 'husky_skateboard':('Skateboard','Push off · Glide'),
 'parkour_stairs_up':('Stair ascent','Staircase'),
 'parkour_stairs_down':('Stair descent','Staircase'),
 'parkour_stairs_up_high':('High-step ascent','High steps'),
 'parkour_stairs_down_high':('High-step descent','High steps'),
 'parkour_slope':('Slope walking','Sloped terrain'),
 'parkour_rough':('Uneven-ground walking','Rough terrain'),
 'parkour_gaps':('Gap crossing','Gaps'),
 'parkour_obstacles':('Obstacle crossing','Obstacle terrain'),
 'pmt_optimized_scene0':('Terrain dance','Uneven ground · Dance walk'),
 'athlete_tennis':('Tennis return','Repeated incoming balls'),
 'roundhouse':('Roundhouse kick','Martial arts'),
 'ultra_largebox':('Box carrying','Two-hand contact · Carry'),
 'mimicx_descent':('Step descent','Step-down motion'),
 'mimicx_bench':('Platform jump','Jump onto a platform'),
 'mimicx_forest':('Beam walking','Narrow-surface balance'),
 'mimicx_football':('Football juggling','Football motion'),
 'mimicx_tennis':('Tennis swing','Racket swing'),
 'mimicx_dance':('Dance','Whole-body dance'),
 'mimicx_kungfu':('Kung fu','Whole-body martial arts'),
}


def presentation_order(tasks):
    """Lead with the featured demos; keep InstinctMJ body tasks at the end."""
    featured = ['husky_skateboard', 'athlete_tennis', 'roundhouse']
    def priority(task):
        if task['id'] in featured:
            return featured.index(task['id'])
        return 4 if task['group'] == 'body' else 3
    return sorted(tasks, key=priority)


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def probe(path):
    d=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate:format=duration','-of','json',str(path)]))
    return float(d['format']['duration']),d['streams'][0]


def finalize_record(task,mode,target,source,report=None,extra=None):
    duration,stream=probe(target)
    poster=target.with_suffix('.webp')
    if not poster.exists():
        subprocess.run(['ffmpeg','-v','error','-ss',str(min(.5,duration/3)),
            '-i',str(target),'-frames:v','1','-quality','91',str(poster)],check=True)
    metadata={'id':f'catalog31/{task}-{mode}','task':task,'expert_enabled':mode=='on',
        'source':str(Path(source).relative_to(ROOT)),'source_sha256':digest(source),
        'duration_s':duration,'fps':30,'speed':1,'resolution':[stream['width'],stream['height']],
        'video_sha256':digest(target),'public_sha256':BASE,'teacher_queries':0,'training_updates':0,
        'full_recorded_episode':True,'pose_interpolation':False,
        'rendering':'Actual closed-loop recording; original speed. Native termination is preserved.'}
    if report:
        metadata.update(report=str(Path(report).relative_to(ROOT)),report_sha256=digest(report))
        d=json.loads(Path(report).read_text())
        if 'visualization' in d:metadata.update(d['visualization'])
        for key in ['seed','plugin','plugin_sha256']:
            if key in d:metadata[key]=d[key]
    metadata.update(extra or {})
    target.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')


def transcode(task,mode,source,report,extra=None,crop_top=0):
    target=MEDIA/f'{task}-{mode}.mp4'
    if target.with_suffix('.json').exists():return
    if not target.exists():
        # The source box video has two provenance header lines above the scene.
        crop=f'crop=iw:ih-{crop_top}:0:{crop_top},' if crop_top else ''
        vf=crop+'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,fps=30'
        subprocess.run(['ffmpeg','-v','error','-i',str(source),'-an','-vf',vf,
            '-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p',
            '-threads','2','-movflags','+faststart',str(target)],check=True)
    finalize_record(task,mode,target,source,report,extra)


def render_native(task,mode):
    folder=WORK/task/mode
    report=folder/'report.json'
    source=folder/'physics.npz'
    if not report.exists():return
    target=MEDIA/f'{task}-{mode}.mp4'
    if target.with_suffix('.json').exists():return
    if not target.exists():
        import numpy as np
        import mujoco
        from studio_lighting import configure_lighting,light_direction
        datafile=np.load(source)
        poses=datafile['qpos'];source_fps=float(datafile['fps'])
        model=mujoco.MjModel.from_binary_path(str(folder/'scene.mjb'))
        configure_lighting(model,'terrain')
        model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
        model.vis.quality.shadowsize=2048
        data=mujoco.MjData(model)
        cam=mujoco.MjvCamera()
        cam.azimuth=55 if task.startswith('parkour_') else 135
        cam.elevation=-22;cam.distance=4 if task.startswith('parkour_') else 3.3
        options=mujoco.MjvOption();options.sitegroup[:]=0;options.geomgroup[3]=0
        pelvis=next(i for i in range(model.nbody) if model.body(i).name=='pelvis' or model.body(i).name.endswith('/pelvis'))
        count=math.ceil(len(poses)/source_fps*30)
        indices=np.minimum(np.rint(np.arange(count)/30*source_fps).astype(int),len(poses)-1)
        process=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-preset','fast',
            '-crf','23','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
        try:
            with mujoco.Renderer(model,height=720,width=1280) as renderer:
                for i in indices:
                    data.qpos[:]=poses[i];mujoco.mj_forward(model,data)
                    cam.lookat[:]=data.xpos[pelvis]
                    cam.lookat[2]=max(-.3,float(cam.lookat[2]))
                    light_direction(model,cam)
                    renderer.update_scene(data,camera=cam,scene_option=options)
                    process.stdin.write(renderer.render().tobytes())
        finally:
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('Encoding failed')
    finalize_record(task,mode,target,source,report,{
        'scene':str((folder/'scene.mjb').relative_to(ROOT)),
        'scene_sha256':digest(folder/'scene.mjb'),
        'rendering':'Nearest actual qpos sample, original speed, preserved native scene and failure. Camera and lighting only; no simulation stepping during rendering.'})
    print('Rendered',task,mode,flush=True)


def group(task):
    if task.startswith(('diveroll_','knee_climb_','jump_sit_','roll_vault_')):return 'body'
    if task in ['husky_skateboard','athlete_tennis','mimicx_football','mimicx_tennis']:return 'sports'
    if task=='ultra_largebox':return 'objects'
    if task in ['roundhouse','mimicx_dance','mimicx_kungfu']:return 'motions'
    return 'terrain'


def exported_record(task,mode):
    if task in LEGACY:
        pairs=json.loads((WEB/'public/data/comparisons.json').read_text())
        pair=pairs[LEGACY[task]]
        key='on' if mode=='on' else 'off'
        return {'id':pair[key+'Id'],'fullId':pair[key+'FullId'],
                'duration':pair[key+'Duration'],'nativeEnd':pair.get(key+'NativeEndS'),
                'continuation':pair.get(key+'ContinuationS',0)}
    bright=WEB/'public/media/gallery-bright-v3'/f'{task}-{mode}.json'
    white=WEB/'public/media/catalog31-white-v2'/f'{task}-{mode}.json'
    sidecar=bright if bright.exists() else white if white.exists() else MEDIA/f'{task}-{mode}.json'
    if not sidecar.exists():return None
    d=json.loads(sidecar.read_text())
    return {'id':d['id'],'fullId':d['id'],'duration':d['duration_s'],
            'nativeEnd':d.get('native_termination_s'),
            'continuation':d.get('post_termination_seconds',0)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--existing',action='store_true');p.add_argument('--native',action='store_true');a=p.parse_args()
    MEDIA.mkdir(parents=True,exist_ok=True)
    analysis=json.loads((ROOT/'reports/paper_teacher_student31_20261106/analysis.json').read_text())
    assert analysis['base_sha256']==BASE and len(analysis['rows'])==31
    if a.existing:
        suite=ROOT/'reports/distillation_suite_v1/run_20261105_173125'
        for row in analysis['rows']:
            task=row['id']
            if task.startswith('mimicx_') or task=='ultra_largebox':
                folder=suite/task/'attempt_1'
                for mode in ['on','off']:
                    role='student' if mode=='on' else 'public'
                    source=folder/(f'final_{role}.mp4' if task=='ultra_largebox' else f'{role}.mp4')
                    report=folder/(f'final_{role}.json' if task=='ultra_largebox' else f'final_{role}_4004.json')
                    if source.exists():transcode(task,mode,source,report,{'resets_after_initialization':0},crop_top=70 if task=='ultra_largebox' else 0)
        source=ROOT/'reports/pmt_target_recovery_long_v1/video/student.mp4'
        if source.exists():transcode('pmt_optimized_scene0','on',source,source.parent/'report.json' if (source.parent/'report.json').exists() else None,crop_top=70)
        source=WORK/'pmt_optimized_scene0/off/native.mp4'
        if (source.parent/'report.json').exists():transcode('pmt_optimized_scene0','off',source,source.parent/'report.json')
    if a.native:
        for row in analysis['rows']:
            for mode in ['on','off']:render_native(row['id'],mode)
    tasks=[]
    for row in analysis['rows']:
        task=row['id'];label,detail=LABELS[task]
        if task in EXCLUDED:continue
        metrics={}
        for role in ['student','public']:
            m=row['metrics'].get(role)
            if m:metrics['on' if role=='student' else 'off']={k:m[k] for k in
                ['n','successes','rate','body_error_m','anchor_error_m'] if k in m}
        item={'id':task,'label':label,'detail':detail,'group':group(task),
              'number':row['index'],'on':exported_record(task,'on'),'off':exported_record(task,'off'),
              'metrics':metrics,'metricLabel':SOURCE_TEXT_EN.get(row['primary_metric'],row['primary_metric']),
              'evidenceNotes':[SOURCE_TEXT_EN.get(note,note) for note in row['notes']],
              'provenance':{'roster_plugin':row['roster_plugin'],
                  'roster_plugin_sha256':row['roster_plugin_sha256'],
                  'sources':row['sources'],'historical_metrics_separate_from_illustrative_video':True}}
        tasks.append(item)
    # This newer native stair task is absent from the original 31-task roster.
    stair_folder=ROOT/'reports/mimicx_stairs_512_v1/run_20261106_083527/training'
    stair_report=stair_folder/'report.json'
    if stair_report.exists():
        d=json.loads(stair_report.read_text())
        assert d['public_sha256']==BASE and d['public_unchanged']
        task='mimicx_stairs'
        plugin=Path(d['selected_plugin'])
        tasks.insert(next(i for i,t in enumerate(tasks) if t['id']=='parkour_stairs_up'),
            {'id':task,'label':'Continuous stair ascent','detail':'Full staircase · Continuous ascent','group':'terrain',
             'number':32,'on':exported_record(task,'on'),'off':exported_record(task,'off'),
             'metrics':{},'metricLabel':'Native continuous-stair scene; one actual recording is shown here.',
             'evidenceNotes':[],
             'provenance':{'roster_plugin':str(plugin.relative_to(ROOT)),
                 'roster_plugin_sha256':digest(plugin),
                 'sources':[str(stair_report.relative_to(ROOT))],
                 'historical_metrics_separate_from_illustrative_video':True}})
    tasks = presentation_order(tasks)
    catalog={'version':2,'groups':GROUPS,'tasks':tasks,
             'public_sha256':BASE,'excluded_base_sha256':analysis['excluded_new_base_sha256'],
             'scope':'Selected legacy task recordings plus the existing native continuous stair task; historical metrics and illustrative replays are separate. No training or new-base results.',
             'excluded_from_display':sorted(EXCLUDED),
             'counts':{'total':len(tasks),'onVideos':sum(t['on'] is not None for t in tasks),
                       'pairedVideos':sum(t['on'] is not None and t['off'] is not None for t in tasks)}}
    path=WEB/'public/data/expert-catalog.json'
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    print('Catalog',catalog['counts'],flush=True)


if __name__=='__main__':main()
