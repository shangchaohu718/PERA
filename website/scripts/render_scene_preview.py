#!/usr/bin/env python3
"""Six-station DESIGN PREVIS, not a continuous policy experiment.

Recorded joint motions are spatially arranged in one visual scene. Walking,
boarding, dismount and pickup joins are animated. No physics rollout or training.
Original experiments/reports/gallery media are never changed.
"""
import argparse,copy,hashlib,json,os,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline,CubicHermiteSpline
from PIL import Image,ImageDraw
from render_gallery import MODEL,studio_model
from render_highlights import CONFIG,load
from studio_lighting import configure_lighting,light_direction
WEB=Path(__file__).resolve().parents[1];ROOT=WEB.parent
WORK=WEB/'.render-work/scene-preview-v1';QA=WEB/'qa/scene-preview';FPS=30
OUT=WORK/'scene-preview.mp4'
SPECS=[('moonwalk',0,4.0,125),('roll',1.9,6.7,110),('skate',0,4.8,75),
       ('rough',.6,2.6,55),('steps',0,3.8,45),('stairs',0,8.98,35),('tennis',0,8.6,125)]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def s5(x):x=np.clip(x,0,1);return x*x*x*(10+x*(-15+6*x))
# Explicit helpers avoid quaternion layout ambiguity.
def rotation(q):return R.from_quat(np.asarray(q)[...,[1,2,3,0]])
def wxyz(r):return r.as_quat()[...,[3,0,1,2]]
def yaw(q):return rotation(q).as_euler('xyz')[...,2]
def arr(v):return ' '.join(f'{float(x):.7g}' for x in np.asarray(v).ravel())
def box(parent,name,pos,size,color,**extra):
 return ET.SubElement(parent,'geom',name=name,type='box',pos=arr(pos),size=arr(size),rgba=arr(color),group='2',contype='0',conaffinity='0',**extra)

def native_ground(m,q):
 d=mujoco.MjData(m);d.qpos[:]=q;mujoco.mj_forward(m,d)
 for i in range(m.ngeom):
  if m.body(int(m.geom_bodyid[i])).name.startswith('robot/') or m.body(int(m.geom_bodyid[i])).name in ('pelvis','torso_link'):m.geom_group[i]=5
 mask=np.array([1,1,1,1,0,0],dtype=np.uint8)
 def height(x,y):
  dist=mujoco.mj_ray(m,d,np.array([x,y,8.]),np.array([0.,0.,-1.]),mask,1,-1,np.zeros(1,dtype=np.int32))
  return 8.-float(dist) if dist>=0 else 0.
 return height

def map_joints(q,m,base):
 result=np.zeros((len(q),36));result[:,:7]=q[:,:7]
 names={m.joint(i).name.split('/')[-1]:int(m.jnt_qposadr[i]) for i in range(m.njnt)} if m is not None else json.loads((WORK/'stairs-layout.json').read_text())['joints']
 for i in range(1,base.njnt):
  name=base.joint(i).name;result[:,int(base.jnt_qposadr[i])]=q[:,names[name]]
 return result

def prepare():
 base=studio_model();clips=[]
 for name,a,b,az in SPECS:
  c=copy.deepcopy(next((c for c in CONFIG['clips'] if c['id']==name),{}))
  if name=='rough':c=dict(id=name,kind='native',source='reports/instinct_parkour_dagger_v1/rough/student_22001/recording/physics.npz',scene='reports/instinct_parkour_dagger_v1/rough/student_22001/recording/scene.mjb',scene_style='terrain')
  if name=='stairs':
   c=dict(id=name,source='reports/mimicx_stairs_512_v1/run_20261106_083527/training/final_student_4004_trajectory.npz',scene='reports/mimicx_stairs_512_v1/run_20261106_083527/training/native_scene.mjb')
   with np.load(ROOT/c['source']) as f:raw=f['qpos'];fps=float(f['fps']);times=np.arange(len(raw))/fps
   m=None
  else:m,raw,times,fps,_=load(c)
  count=round((b-a)*FPS);indices=np.clip(np.rint((a+np.arange(count)/FPS)*fps).astype(int),0,len(raw)-1)
  sample_times=a+np.arange(count)/FPS
  source_times=np.arange(len(raw))/fps
  selected=CubicSpline(source_times,raw,axis=0)(sample_times)
  free=[int(m.jnt_qposadr[i]) for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE] if m is not None else [0]
  for adr in free:selected[:,adr+3:adr+7]=wxyz(Slerp(source_times,rotation(raw[:,adr+3:adr+7]))(sample_times))
  q=map_joints(selected,m,base)
  th=-float(yaw(q[0,3:7]));rot=R.from_euler('z',th);origin=q[0,:3].copy()
  ground=native_ground(m,selected[0]) if name in ('rough','steps') else lambda x,y:0.
  floor0=ground(*origin[:2]);floor1=ground(*q[-1,:2])
  if name=='stairs':floor1=max(step['pos'][2]+step['size'][2] for step in json.loads((WORK/'stairs-layout.json').read_text())['steps'])
  local=rot.apply(q[:,:3]-[origin[0],origin[1],floor0]);q[:,:3]=local;q[:,3:7]=wxyz(rot*rotation(q[:,3:7]))
  clip=dict(id=name,q=q,az=az,source=c['source'],scene=c.get('scene'),source_sha256=sha(ROOT/c['source']),source_fps=fps,source_interval=[a,b],origin=origin,rot=rot,floor0=floor0,floor_gain=floor1-floor0,model=m,ground=ground,raw=selected,offset=np.zeros(3))
  if name=='skate':
   # Deck tilt body carries the actual recorded free-body and tilt transforms.
   d=mujoco.MjData(m);poses=[]
   for pose in selected:
    d.qpos[:]=pose;mujoco.mj_forward(m,d);i=m.body('skateboard/board_tilt_body').id
    poses.append([*d.xpos[i],*d.xquat[i]])
   poses=np.asarray(poses);poses[:,:3]=rot.apply(poses[:,:3]-[origin[0],origin[1],floor0]);poses[:,3:7]=wxyz(rot*rotation(poses[:,3:7]));clip['board']=poses
  if name=='tennis':
   d=mujoco.MjData(m);balls=[]
   for pose in selected:
    d.qpos[:]=pose;mujoco.mj_forward(m,d);balls.append(d.xpos[m.body('tennis_ball').id].copy())
   clip['ball']=rot.apply(np.array(balls)-[origin[0],origin[1],floor0])
  clips.append(clip)
 # A measured walking sequence supplies gait shape; joins remain explicitly animated.
 source=ROOT/'reports/velocity_course_v11/seed7002/physical_replay.npz'
 with np.load(source) as f:
  ids=np.flatnonzero((f['mode']=='navigate')&(f['task']=='diveroll_lab_no_wall'))
  raw=f['qpos'][ids[25:100]].copy()
 m=mujoco.MjModel.from_binary_path(str(ROOT/'reports/velocity_course_v11/seed7002/scene.mjb'))
 walk=map_joints(raw,m,base)
 return base,clips,walk

def bridge(a,b,walk,count):
 D=count/FPS;t=np.arange(1,count)/FPS;u=t/D;count=len(t)
 p0=a[-1,:3];p1=b[0,:3];v0=(a[-1,:3]-a[-3,:3])*FPS/2;v1=(b[2,:3]-b[0,:3])*FPS/2
 v0[2]=v1[2]=0
 h00=2*u**3-3*u**2+1;h10=u**3-2*u**2+u;h01=-2*u**3+3*u**2;h11=u**3-u**2
 q=np.zeros((count,36));q[:,:3]=h00[:,None]*p0+h10[:,None]*D*v0+h01[:,None]*p1+h11[:,None]*D*v1
 q[:,2]=p0[2]+s5(u)*(p1[2]-p0[2])
 w=walk[np.arange(count)*25//FPS%len(walk),7:].copy()
 e0=a[-1,7:];e1=b[0,7:];jv0=(a[-1,7:]-a[-2,7:])*FPS;jv1=(b[1,7:]-b[0,7:])*FPS
 first=s5(t/.45)[:,None];last=s5((D-t)/.45)[:,None]
 q[:,7:]=(1-first)*(e0+np.clip(t,0,.3)[:,None]*jv0)+first*w
 q[:,7:]=last*q[:,7:]+(1-last)*(e1+np.clip(t-D,-.3,0)[:,None]*jv1)
 slerp=Slerp([0,1],rotation(np.array([a[-1,3:7],b[0,3:7]])))
 q[:,3:7]=wxyz(slerp(s5(u)))
 return q

def timeline(clips,walk):
 chunks=[];labels=[];az=[];rows=[];joins=[];cursor=0;level=0.;last=None
 for c in clips:
  q=c['q'].copy()
  if last is None:offset=np.array([0.,0.,0.])
  else:
   D=2.5 if c['id']=='tennis' else 1.5;speed0=float(np.linalg.norm((last[-1,:2]-last[-3,:2])*FPS/2));speed1=float(np.linalg.norm((q[2,:2]-q[0,:2])*FPS/2))
   distance=max(.85,D*(speed0+speed1)/2+.25)
   # Native endpoints are spatially arranged, never called a policy handoff.
   offset=np.array([last[-1,0]+distance,0.,level])
  if c['id']=='rough':offset[2]+=.23
  q[:,:3]+=offset;c['offset']=offset
  if last is not None:
   join=bridge(last,q,walk,round(D*FPS))
   if c['id']=='tennis':
    t=np.arange(1,round(D*FPS))/FPS
    waypoint=q[0,:2]+[-.48,-.30]
    v0=(last[-1,:2]-last[-3,:2])*FPS/2;v1=(q[2,:2]-q[0,:2])*FPS/2
    join[:,:2]=CubicHermiteSpline([0,D-.9,D],np.array([last[-1,:2],waypoint,q[0,:2]]),np.array([v0,[.18,0.],v1]))(t)
   start=cursor/FPS;chunks.append(join);labels+=['animated_join']*len(join)
   az.extend((az[-1]+s5(np.arange(len(join))/len(join))*(c['az']-az[-1])).tolist())
   cursor+=len(join);joins.append(dict(start_s=start,end_s=cursor/FPS,kind='animated locomotion / boarding / dismount / pickup previsualization'))
  c['start_frame']=cursor;c['end_frame']=cursor+len(q);c['world_q']=q
  chunks.append(q);labels+=[c['id']]*len(q);az.extend([c['az']]*len(q));cursor+=len(q)
  rows.append(dict(id=c['id'],start_s=c['start_frame']/FPS,end_s=c['end_frame']/FPS,source=c['source'],source_sha256=c['source_sha256'],source_interval_s=c['source_interval'],spatial_rotation_rad=float(c['rot'].as_euler('xyz')[2]),spatial_offset=offset.tolist()))
  level=offset[2]+c['floor_gain'];last=q
 q=np.concatenate(chunks);endcount=30
 # Animated final settling is part of this explicitly declared design preview.
 rest=q[-1].copy()
 end=np.repeat(q[-1][None],endcount,axis=0);mix=s5(np.linspace(0,1,endcount))[:,None];end[:,7:]=end[:,7:]*(1-mix)+rest[None,7:]*mix
 q=np.concatenate([q,end]);labels+=['animated_settle']*endcount;az.extend([az[-1]]*endcount)
 keys=[(0,125),(4,125),(9,110),(16.5,75),(22,55),(29,35),(39,125),(len(q)/FPS,125)]
 time=np.arange(len(q))/FPS;camera=np.zeros(len(q))
 for (ta,va),(tb,vb) in zip(keys[:-1],keys[1:]):
  mask=(time>=ta)&(time<=tb);camera[mask]=va+s5((time[mask]-ta)/(tb-ta))*(vb-va)
 return q,np.array(labels),camera,rows,joins

def terrain_mesh(asset,world,c):
 q=c['raw'];low=q[:,:2].min(0)-[.65,.85];high=q[:,:2].max(0)+[.65,.85]
 xs=np.linspace(low[0],high[0],max(12,round((high[0]-low[0])/.07)));ys=np.linspace(low[1],high[1],max(12,round((high[1]-low[1])/.07)))
 points=np.array([[x,y,c['ground'](x,y)] for y in ys for x in xs]);points=c['rot'].apply(points-[*c['origin'][:2],c['floor0']])+c['offset']
 faces=[];W=len(xs)
 for iy in range(len(ys)-1):
  for ix in range(W-1):
   i=iy*W+ix;faces.extend([[i,i+1,i+W],[i+1,i+W+1,i+W]])
 name='preview_'+c['id']+'_terrain';ET.SubElement(asset,'mesh',name=name,vertex=arr(points),face=' '.join(map(str,np.asarray(faces).ravel())))
 ET.SubElement(world,'geom',name=name,type='mesh',mesh=name,rgba='.66 .72 .79 1',group='2',contype='0',conaffinity='0')
 c['terrain_bounds']=[points.min(0).tolist(),points.max(0).tolist()]

def build_scene(clips):
 tree=ET.parse(MODEL);root=tree.getroot();root.find('compiler').set('meshdir',str((MODEL.parent/'../meshes').resolve()));asset=root.find('asset');world=root.findall('worldbody')[-1]
 ET.SubElement(root,'visual') if root.find('visual') is None else None
 for tex in root.findall('.//texture'):
  if tex.get('type')=='skybox':tex.set('rgb1','1 1 1');tex.set('rgb2','1 1 1')
 for mat in root.findall('.//material'):
  if mat.get('name')=='groundplane':mat.attrib.pop('texture',None);mat.set('rgba','1 1 1 1')
 for g in root.findall('.//geom[@name="floor"]'):g.set('group','2')
 ET.SubElement(world,'light',name='preview_fill',directional='true',dir='0.2 0.2 -1',castshadow='false')
 # World-anchored grid and a quiet blue route edge convey displacement.
 xmax=float(clips[-1]['world_q'][-1,0])+12
 for x in np.arange(-3,xmax+2,1.):box(world,'grid_x_'+str(round(x,2)),[x,0,.001],[.003,4,.001],[.90,.92,.94,1])
 for y in np.arange(-4,4.1,1.):box(world,'grid_y_'+str(round(y,2)),[xmax/2,y,.001],[xmax/2+3,.003,.001],[.90,.92,.94,1])
 box(world,'route_edge',[xmax/2,-1.7,.005],[xmax/2+2,.016,.004],[.50,.70,.91,1])
 # Each stage remains in this same scene for the entire shot.
 for c in clips:
  if c['id'] in ('rough','steps'):terrain_mesh(asset,world,c)
  if c['id']=='stairs':
   info=json.loads((WORK/'stairs-layout.json').read_text())
   for i,step in enumerate(info['steps']):
    pos=c['rot'].apply(np.array(step['pos'])-[*c['origin'][:2],c['floor0']])+c['offset'];size=step['size']
    box(world,f'preview_stair_{i}',pos,size,[.59,.67,.77,1])
   ground=c['offset'][2]
   if ground>.02:box(world,'stairs_support',[c['offset'][0]+2.8,0,ground/2-.01],[3.9,1.4,ground/2],[.76,.81,.87,1])
 tennis=clips[-1];floor=tennis['offset'][2];cx=float(tennis['world_q'][0,0])+5.5
 if floor>.01:box(world,'court_platform',[cx,0,floor/2-.01],[6.1,4.5,floor/2],[.76,.81,.87,1])
 box(world,'court_surface',[cx,0,floor+.003],[6.1,4.5,.003],[.18,.45,.50,1])
 # Court lines are visible close to the player, including a central service line.
 for i,x in enumerate([cx-5.9,cx-2.8,cx+5.8]):box(world,f'court_line_x_{i}',[x,0,floor+.009],[.02,4.1,.003],[.94,.97,.98,1])
 for i,y in enumerate([-4.1,0,4.1]):box(world,f'court_line_y_{i}',[cx,y,floor+.009],[5.9,.018,.003],[.94,.97,.98,1])
 # A short entry ramp and grounded edges connect the raised uneven lane.
 rough=next(c for c in clips if c['id']=='rough');skate=next(c for c in clips if c['id']=='skate')
 x0=float(skate['world_q'][-1,0])+.5;x1=float(rough['world_q'][0,0])-.1;h=float(rough['offset'][2])
 if x1>x0:
  v=np.array([[x0,-.85,-.01],[x0,.85,-.01],[x1,-.85,-.01],[x1,.85,-.01],[x1,-.85,h],[x1,.85,h]])
  faces=np.array([[0,2,4],[1,5,3],[0,4,5],[0,5,1],[2,3,5],[2,5,4],[0,1,3],[0,3,2]])
  ET.SubElement(asset,'mesh',name='preview_entry_ramp',vertex=arr(v),face=' '.join(map(str,faces.ravel())))
  ET.SubElement(world,'geom',name='preview_entry_ramp',type='mesh',mesh='preview_entry_ramp',rgba='.66 .72 .79 1',group='2',contype='0',conaffinity='0')
 # Racket rack: its support meets the recorded wrist location at the pickup pose.
 bm=studio_model();bd=mujoco.MjData(bm);bd.qpos[:]=tennis['world_q'][0];mujoco.mj_forward(bm,bd)
 wp=bd.xpos[bm.body('right_wrist_yaw_link').id].copy()+[-.65,-.40,0]
 box(world,'racket_rack_base',[wp[0]+.04,wp[1],floor+.035],[.19,.2,.035],[.62,.69,.76,1])
 box(world,'racket_rack_stem',[wp[0]+.15,wp[1],(floor+wp[2])*.5],[.025,.025,(wp[2]-floor)*.5],[.62,.69,.76,1])
 box(world,'racket_rack_top',[wp[0]+.09,wp[1],wp[2]-.025],[.1,.1,.018],[.62,.69,.76,1])
 # Net at the far half of the practice court.
 nx=cx
 for j,ny in enumerate([-4.,4.]):
  ET.SubElement(world,'geom',name=f'court_post_{j}',type='capsule',fromto=arr([nx,ny,floor,nx,ny,floor+.92]),size='.025',rgba='.40 .46 .52 1',group='2',contype='0',conaffinity='0')
 ET.SubElement(world,'geom',name='court_net_band',type='capsule',fromto=arr([nx,-4,floor+.9,nx,4,floor+.9]),size='.012',rgba='.94 .96 .98 1',group='2',contype='0',conaffinity='0')
 box(world,'court_net',[nx,0,floor+.48],[.008,4,.4],[.30,.36,.41,.18])
 # The skateboard is a visual stand-in driven by actual recorded deck poses.
 body=ET.SubElement(world,'body',name='preview_board',mocap='true',pos='0 0 -3')
 box(body,'preview_board_deck',[0,0,0],[.4,.1,.01],[.18,.24,.31,1])
 box(body,'preview_board_grip',[-.2,0,.011],[.15,.075,.002],[.29,.34,.40,1])
 for j,x in enumerate([-.26,.26]):
  box(body,f'preview_truck_{j}',[x,0,-.035],[.045,.1,.025],[.32,.37,.42,1])
  ET.SubElement(body,'geom',name=f'preview_axle_{j}',type='capsule',fromto=arr([x,-.16,-.065,x,.16,-.065]),size='.008',rgba='.40 .46 .52 1',group='2',contype='0',conaffinity='0')
 for i,(x,y) in enumerate([(-.26,-.16),(-.26,.16),(.26,-.16),(.26,.16)]):
  ET.SubElement(body,'geom',name=f'preview_wheel_{i}',type='cylinder',size='.035 .022',pos=arr([x,y,-.065]),quat='.7071068 .7071068 0 0',rgba='.12 .15 .19 1',group='2',contype='0',conaffinity='0')
 ball=ET.SubElement(world,'body',name='preview_ball',mocap='true',pos='0 0 -3');ET.SubElement(ball,'geom',name='preview_ball_geom',type='sphere',size='.0335',rgba='.82 .96 .07 1',group='2',contype='0',conaffinity='0')
 athlete=ROOT/'third_party/ATHLETE/robots/replay_unitree_description/mjcf/g1.xml';aroot=ET.parse(athlete).getroot();mesh=copy.deepcopy(aroot.find('.//mesh[@name="racket_mesh"]'));mesh.set('name','preview_racket_mesh');mesh.set('file',str(athlete.parent.parent/'meshes/racket.obj'));asset.append(mesh)
 racket=copy.deepcopy(aroot.find('.//geom[@name="racket_visual"]'));racket.set('mesh','preview_racket_mesh');racket.attrib.pop('material',None);racket.set('rgba','.12 .14 .17 1');racket.set('name','preview_racket_attached')
 root.find('.//body[@name="right_wrist_yaw_link"]').append(racket)
 rack=ET.SubElement(world,'body',name='preview_racket_parked',mocap='true',pos='0 0 -3');g=copy.deepcopy(racket);g.set('name','preview_racket_waiting');rack.append(g)
 path=WORK/'scene.xml';tree.write(path)
 model=configure_lighting(mujoco.MjModel.from_xml_path(str(path)))
 # Preserve all palette choices: studio configuration only adjusts its own floor.
 for i in range(model.ngeom):
  if model.geom(i).name.startswith('preview_') and model.geom_bodyid[i]==0:pass
 model.vis.map.zfar=80;model.stat.extent=25
 return model

def board_path(clips,count):
 c=next(c for c in clips if c['id']=='skate');p=c['board'].copy();p[:,:3]+=c['offset'];out=np.repeat(p[0][None],count,axis=0);out[c['start_frame']:c['end_frame']]=p
 v=(p[-1,:3]-p[-3,:3])*FPS/2
 for i in range(c['end_frame'],count):
  dt=min((i-c['end_frame'])/FPS,.7);out[i]=p[-1];out[i,:3]+=v*(dt-dt*dt/(2*.7))
 return out

def render(preview=False):
 WORK.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
 base,clips,walk=prepare();q,labels,az,rows,joins=timeline(clips,walk);model=build_scene(clips);data=mujoco.MjData(model)
 assert np.isfinite(q).all() and np.max(abs(np.linalg.norm(q[:,3:7],axis=1)-1))<1e-5
 boundaries=[c['start_frame'] for c in clips[1:]]+[c['end_frame'] for c in clips[:-1]]
 checks={'camera_cuts':0,'duration_s':len(q)/FPS,'max_camera_angular_speed_deg_s':float(np.max(abs(np.diff(az)))*FPS),'max_root_step_at_join_m':float(max(np.linalg.norm(q[i,:3]-q[i-1,:3]) for i in boundaries)),'max_joint_step_at_join_rad':float(max(np.max(abs(q[i,7:]-q[i-1,7:])) for i in boundaries))}
 assert checks['max_root_step_at_join_m']<.12 and checks['max_camera_angular_speed_deg_s']<22
 (QA/'continuity-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 board=board_path(clips,len(q));tennis=clips[-1]
 # Match the parked and held racket at the same wrist transform at pickup.
 data.qpos[:36]=tennis['world_q'][0];mujoco.mj_forward(model,data);wrist=model.body('right_wrist_yaw_link').id
 rack_target=data.xpos[wrist].copy()+[-.65,-.40,0]
 pickup=tennis['start_frame']-27
 arm=[model.joint(name).id for name in ('right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint')]
 qa=np.array([int(model.jnt_qposadr[j]) for j in arm]);da=np.array([int(model.jnt_dofadr[j]) for j in arm])
 data.qpos[:36]=q[pickup];original=q[pickup,qa].copy()
 for _ in range(45):
  mujoco.mj_forward(model,data);error=rack_target-data.xpos[wrist]
  if np.linalg.norm(error)<.002:break
  jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mujoco.mj_jacBody(model,data,jp,jr,wrist);J=jp[:,da]
  delta=J.T@np.linalg.solve(J@J.T+.001*np.eye(3),error)
  data.qpos[qa]+=np.clip(delta,-.12,.12)
  for j,adr in zip(arm,qa):
   if model.jnt_limited[j]:data.qpos[adr]=np.clip(data.qpos[adr],*model.jnt_range[j])
 change=data.qpos[qa]-original
 for i in range(max(0,pickup-20),min(len(q),pickup+27)):
  width=20 if i<=pickup else 27
  weight=.5+.5*np.cos(np.pi*abs(i-pickup)/width);q[i,qa]+=weight*change
 data.qpos[:36]=q[pickup];mujoco.mj_forward(model,data)
 rack_pos=data.xpos[wrist].copy();rack_quat=data.xquat[wrist].copy()
 checks['racket_pickup_wrist_error_m']=float(np.linalg.norm(rack_target-rack_pos))
 assert checks['racket_pickup_wrist_error_m']<.03,checks
 (QA/'continuity-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 attached=model.geom('preview_racket_attached').id;waiting=model.geom('preview_racket_waiting').id
 # Camera filtering has no effect on recorded or animated robot poses.
 target=q[:,:3].copy();target[:,0]+=.25;target[:,2]-=.07
 roll=next(c for c in clips if c['id']=='roll');target[roll['start_frame']:roll['end_frame'],2]=roll['offset'][2]+.65
 target=gaussian_filter1d(target,FPS*.20,axis=0,mode='nearest')
 distance=np.full(len(q),3.65);distance[labels=='roll']=4.1;distance[labels=='skate']=3.9
 distance[tennis['start_frame']:]=3.4;distance=gaussian_filter1d(distance,FPS*.35)
 cam=mujoco.MjvCamera();cam.elevation=-23
 options=mujoco.MjvOption();options.sitegroup[:]=0;options.geomgroup[0]=0
 selected=np.unique(np.rint(np.linspace(0,len(q)-1,18)).astype(int)) if preview else range(len(q))
 proc=None;frames=[]
 if not preview:
  if OUT.exists():raise FileExistsError(OUT)
  proc=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','slow','-crf','21','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(OUT)],stdin=subprocess.PIPE)
 try:
  with mujoco.Renderer(model,height=1080,width=1920) as renderer:
   for i in selected:
    data.qpos[:36]=q[i];data.mocap_pos[model.body('preview_board').mocapid[0]]=board[i,:3];data.mocap_quat[model.body('preview_board').mocapid[0]]=board[i,3:7]
    bm=model.body('preview_ball').mocapid[0];data.mocap_pos[bm]=tennis['ball'][min(max(i-tennis['start_frame'],0),len(tennis['ball'])-1)]+tennis['offset'] if i>=tennis['start_frame'] else [0,0,-3]
    rm=model.body('preview_racket_parked').mocapid[0];data.mocap_pos[rm]=rack_pos;data.mocap_quat[rm]=rack_quat
    model.geom_rgba[attached,3]=1 if i>=pickup else 0;model.geom_rgba[waiting,3]=0 if i>=pickup else 1
    mujoco.mj_forward(model,data);cam.lookat[:]=target[i];cam.azimuth=float(az[i]);cam.distance=float(distance[i]);light_direction(model,cam)
    renderer.update_scene(data,camera=cam,scene_option=options);frame=renderer.render()
    if preview:frames.append((i,Image.fromarray(frame).resize((480,270))))
    else:
     proc.stdin.write(frame.tobytes())
     if i==30:Image.fromarray(frame).save(WORK/'scene-preview.webp',quality=93)
     if i%180==0:print(f'{i}/{len(q)}',flush=True)
 finally:
  if proc:proc.stdin.close();assert proc.wait()==0
 if preview:
  sheet=Image.new('RGB',(480*3,300*6),'white');draw=ImageDraw.Draw(sheet)
  for j,(i,f) in enumerate(frames):
   x=(j%3)*480;y=(j//3)*300;sheet.paste(f,(x,y));draw.text((x+12,y+275),f'{i/FPS:.2f}s  {labels[i]}',fill='black')
  sheet.save(QA/'contact-sheet.jpg',quality=92)
 else:
  sources=[dict(path=r['source'],sha256=r['source_sha256']) for r in rows]
  sources.append(dict(path='reports/velocity_course_v11/seed7002/physical_replay.npz',sha256=sha(ROOT/'reports/velocity_course_v11/seed7002/physical_replay.npz')))
  meta=dict(id='scene-preview',name='Six-station scene and camera previsualization',design_previsualization=True,policy_executed=False,continuous_policy_rollout=False,continuous_episode=False,single_rendered_scene=True,camera_cuts=0,animated_transitions=True,spatially_retargeted_recorded_motions=True,pose_interpolation=True,speed=1,burned_in_labels=False,fps=FPS,resolution=[1920,1080],output_frames=len(q),duration_s=len(q)/FPS,video_sha256=sha(OUT),sources=sources,stages=rows,animated_joins=joins,animated_settle_s=1,scene_sha256=sha(WORK/'scene.xml'),scope='DESIGN PREVIS ONLY. Recorded joint motions are spatially arranged in one scene. Walking/boarding/dismount/pickup joins and final settling are animated; terrain/board/court are presentation geometry. No policy executes in this film. It is not evidence of one-policy continuous physics, mounting, grasping, contact accuracy or success rate. The experiment gallery remains separate and unchanged.')
  (WORK/'scene-preview.json').write_text(json.dumps(meta,indent=2)+'\n');np.savez_compressed(WORK/'timeline.npz',qpos=q,labels=labels,camera_target=target,camera_azimuth=az,camera_distance=distance,board=board,fps=FPS)
  print('Complete',len(q)/FPS,flush=True)
 # Save the route independently of the render for review and future physical integration.
 (WORK/'route.json').write_text(json.dumps(dict(stages=rows,joins=joins),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args();render(args.preview)
