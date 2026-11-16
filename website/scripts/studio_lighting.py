"""Shared studio lighting for faithful G1 state re-renders."""
import mujoco
import numpy as np

def configure_lighting(model, scene_style='studio'):
    """Camera fill plus a directional key and a shadow-free rim/fill.

    Preserve robot albedo and all geometry. Brightness comes from actual lights,
    not a post-production exposure filter. A white studio floor retains shadows.
    """
    model.vis.global_.offwidth=1920;model.vis.global_.offheight=1080
    model.vis.quality.shadowsize=4096
    model.vis.headlight.active=1
    model.vis.headlight.ambient[:]=.48
    model.vis.headlight.diffuse[:]=.65
    model.vis.headlight.specular[:]=.12
    model.vis.rgba.haze[:]=[1,1,1,1]
    for i in range(model.nlight):
        model.light_ambient[i]=0
        model.light_diffuse[i]=.32 if i==0 else .18
        model.light_specular[i]=.10 if i==0 else .04
        model.light_type[i]=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        model.light_castshadow[i]=i==0
    for i in range(model.nmat):
        name=model.mat(i).name
        if 'groundplane' in name:
            model.mat_emission[i]=.14 if scene_style=='studio' else 0
            model.mat_reflectance[i]=0
            model.mat_rgba[i]=[1,1,1,1] if scene_style=='studio' else [.50,.55,.61,1] if scene_style=='terrain' else [.72,.73,.74,1]
            if scene_style=='skate':model.mat_texrepeat[i]=[1,1]
        if name.startswith('tennis_court/'):
            model.mat_reflectance[i]=0
            # Keep the source court colors, white markings, net and yellow ball.
            model.mat_emission[i]=0
    for i in range(model.ngeom):
        name=model.geom(i).name
        body=model.body(int(model.geom_bodyid[i])).name
        if (name in ('floor','terrain') or body=='terrain') and model.geom_rgba[i,3]>0:
            model.geom_rgba[i]=[1,1,1,1]
        if scene_style=='terrain' and body=='terrain':
            if model.geom_matid[i]<0:model.geom_rgba[i]=[.50,.55,.61,1]
        elif scene_style=='studio' and body=='terrain' and name not in ('terrain','floor') and model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_PLANE:
            # Keep raised obstacles readable against the white backdrop.
            if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH or model.geom_size[i,2]>.055:
                model.geom_matid[i]=-1
                model.geom_rgba[i]=[.78,.81,.85,1]
    for i in range(model.ntex):
        address=model.tex_adr[i];h=int(model.tex_height[i]);w=int(model.tex_width[i]);n=int(model.tex_nchannel[i])
        pixels=model.tex_data[address:address+h*w*n].reshape(h,w,n)
        if model.tex_type[i]==mujoco.mjtTexture.mjTEXTURE_SKYBOX:pixels[:]=255
        elif scene_style=='tennis':
            continue  # Actual green court surround texture.
        elif scene_style=='skate':
            pixels[:]=[212,216,221][:n]
            # A world-anchored ground grid makes displacement easy to see.
            for offset in (0,1):
                pixels[offset::max(1,h//2),:,:3]=[142,153,168]
                pixels[:,offset::max(1,w//2),:3]=[142,153,168]
        elif scene_style=='terrain':
            pixels[:]=[155,169,186][:n]
            pixels[::75,:,:3]=[135,149,166];pixels[:,::75,:3]=[135,149,166]
        else:
            pixels[:]=255
            pixels[::75,:,:3]=253;pixels[:,::75,:3]=253
    return model


def light_direction(model,camera):
    for i in range(model.nlight):
        angle=np.radians(camera.azimuth+(35 if i==0 else 155))
        model.light_dir[i]=[-.45*np.cos(angle),-.45*np.sin(angle),-1]

