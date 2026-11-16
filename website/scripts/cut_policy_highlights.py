"""Four clearly edited highlights from the audited physical course film."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

WEB=Path(__file__).resolve().parents[1]
SELECTION=[('skateboard',0,144),('rough_ground_and_steps',368,524),('upstairs',570,762),('tennis',905,1205)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    source=args.source;out=args.output;out.mkdir(parents=True,exist_ok=False)
    full=json.loads(source.with_suffix('.json').read_text());assert full['physical_policy_course'] and full['pose_interpolation'] is False
    assert sha(source)==full['video_sha256']
    filters=[];chapters=[];cursor=0
    for i,(name,a,b) in enumerate(SELECTION):
        filters.append(f'[0:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v{i}]')
        chapters.append(dict(id=name,start_s=cursor/30,end_s=(cursor+b-a)/30,source_start_frame=a,source_end_frame=b,source_start_s=a/30,source_end_s=b/30))
        cursor+=b-a
    filters.append(''.join(f'[v{i}]' for i in range(len(SELECTION)))+'concat=n=4:v=1:a=0[video]')
    video=out/'scene-preview.mp4'
    subprocess.run(['ffmpeg','-v','error','-i',str(source),'-filter_complex',';'.join(filters),'-map','[video]','-an','-c:v','libx264','-preset','slow','-crf','21','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    subprocess.run(['ffmpeg','-v','error','-ss','1','-i',str(video),'-frames:v','1','-quality','93',str(out/'scene-preview.webp')],check=True)
    meta=copy.deepcopy(full)
    meta.update(id='scene-preview',name='Four scenes · recorded policy highlights',physical_policy_course=False,policy_course_highlights=True,
                edited_montage=True,continuous_episode=False,continuous_policy_rollout=False,selected_excerpts=True,
                full_video='policy-course-full.mp4',full_video_sha256=full['video_sha256'],source_recording_continuous=True,
                chapters=chapters,output_frames=cursor,duration_s=cursor/30,video_sha256=sha(video),camera_cuts=3,
                scope='Selected original-speed excerpts from one audited physical policy rollout, with hard cuts between four scenes. No animated transitions or robot pose editing. The tennis excerpt shows the first two incoming-ball cycles, both with recorded racket contact; the complete source also retains a third miss. A shared frozen public Actor executes all stages with explicitly selected experts. The full source and physical replay audit remain available.')
    (out/'scene-preview.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Encoded',cursor,'frames /',cursor/30,'seconds',flush=True)
if __name__=='__main__':main()
