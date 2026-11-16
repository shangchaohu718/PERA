"""Compact skill-region presentation; policy state and weights are untouched."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compact_child_spec(child, row, index, *, bit_offset=None):
    """Relabel native mask bits without changing any within-region contact pair."""
    offset = 3 * index if bit_offset is None else bit_offset
    bounds = {}
    if row["kind"] == "vault":
        native = child.compile()
        data = mujoco.MjData(native)
        mujoco.mj_forward(native, data)
        for i in range(native.ngeom):
            if native.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = native.geom_dataid[i]
            start, count = native.mesh_vertadr[mesh], native.mesh_vertnum[mesh]
            vertices = (
                native.mesh_vert[start : start + count] @ data.geom_xmat[i].reshape(3, 3).T
                + data.geom_xpos[i]
            )
            bounds[native.geom(i).name] = (vertices.min(0), vertices.max(0))
    union = 0
    for geom in child.geoms:
        union |= geom.contype | geom.conaffinity
    bits = [i for i in range(31) if union & (1 << i)]
    if offset + len(bits) > 31:
        raise ValueError("Too many independent collision-mask bits")

    def remap(mask):
        return sum(1 << (offset + j) for j, bit in enumerate(bits) if mask & (1 << bit))

    for geom in child.geoms:
        geom.contype = remap(geom.contype)
        geom.conaffinity = remap(geom.conaffinity)
        name = geom.name or ""
        if row["kind"] == "tennis" and name.startswith("tennis_court/"):
            # Crop only noncolliding decorative boxes; preserve the net collider.
            if (
                geom.contype == 0
                and geom.conaffinity == 0
                and geom.type == mujoco.mjtGeom.mjGEOM_BOX
            ):
                pos, size = np.array(geom.pos), np.array(geom.size)
                lo = np.maximum(pos[:2] - size[:2], [-2.0, -3.0])
                hi = np.minimum(pos[:2] + size[:2], [8.5, 3.0])
                if np.any(hi <= lo):
                    color = np.array(geom.rgba)
                    color[3] = 0
                    geom.rgba = color
                else:
                    pos[:2], size[:2] = (lo + hi) / 2, (hi - lo) / 2
                    geom.pos, geom.size = pos, size
        if row["kind"] == "vault" and name.startswith("terrain_"):
            # Far background hulls stay in physics but are hidden in the overview.
            lo, hi = bounds[geom.name]
            if (lo[1] + hi[1]) / 2 < -3.5 or hi[2] > 1.05:
                color = np.array(geom.rgba)
                color[3] = 0
                geom.rgba = color
            elif hi[2] > 0.2 and lo[0] < 0.55 and hi[0] > -0.45 and lo[1] < -0.9 and hi[1] > -1.85:
                geom.material = ""
                geom.rgba = [0.85, 0.55, 0.25, 1]

    return len(bits)


def project(scene, point, width, height):
    camera = scene.camera[0]
    p = np.asarray(point) - camera.pos
    forward, up = np.array(camera.forward), np.array(camera.up)
    right = np.cross(forward, up)
    z = float(p @ forward)
    if z <= 0:
        return None
    near = float(camera.frustum_near)
    fy = float(camera.frustum_top - camera.frustum_bottom)
    fx = fy * width / height
    x = (float(p @ right) * near / z - float(camera.frustum_center)) / fx + 0.5
    y = (float(p @ up) * near / z - float(camera.frustum_bottom)) / fy
    return round(x * width), round((1 - y) * height)


def render_frame(renderer, arena):
    clean = arena.config.get("clean_presentation", False)
    top = 0 if clean else 64
    frame = Image.new("RGB", (1920, 900 if clean else 1080), (16, 24, 34))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
    small = ImageFont.truetype("DejaVuSans.ttf", 18)
    tiny = ImageFont.truetype("DejaVuSans.ttf", 16)
    if not clean:
        draw.text(
            (16, 10),
            f"PERA  |  ONE SHARED HOLOMOTION ACTOR  |  {len(arena.slots)} EXPERTS  |  ONE PHYSICS WORLD",
            font=font,
            fill="white",
        )
        draw.text(
            (16, 38),
            f"Actual student control  |  {arena.data.time:.2f}s  |  {len(arena.reset_events)} labeled episode resets",
            font=small,
            fill="#bdd0df",
        )
    camera = mujoco.MjvCamera()
    view = arena.config["overview_camera"]
    camera.lookat[:] = view["lookat"]
    camera.distance = view["distance"]
    camera.azimuth = view.get("azimuth", 90)
    camera.elevation = view.get("elevation", -65)
    renderer.update_scene(arena.data, camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    main = Image.fromarray(renderer.render().copy()).resize((1440, 900))
    md = ImageDraw.Draw(main)
    for i, slot in enumerate(arena.slots):
        point = arena.data.xpos[slot.map.bmap[slot.pelvis]].copy()
        point[2] += 0.8 if clean else 1.15
        xy = project(renderer.scene, point, 1440, 900)
        if xy and 0 <= xy[0] < 1440 and 0 <= xy[1] < 900:
            label = f"{i + 1}. {slot.row['label']}"
            if clean and slot.episode > 1 and slot.episode_step <= 15:
                label += " / Reset"
            box = md.textbbox(xy, label, font=tiny)
            md.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), fill=(16, 24, 34))
            md.text(xy, label, font=tiny, fill="#ff8a80" if slot.lost else "white")
    frame.paste(main, (0, top))
    groups = (len(arena.slots) + 2) // 3
    group = int(arena.data.time // 4) % groups
    ids = [(group * 3 + j) % len(arena.slots) for j in range(3)]
    for j, i in enumerate(ids):
        slot = arena.slots[i]
        pelvis = arena.data.xpos[slot.map.bmap[slot.pelvis]].copy()
        camera.lookat[:] = pelvis + [0, 0, 0.15]
        camera.distance = slot.row.get("close_distance", 3.7)
        camera.azimuth, camera.elevation = (
            slot.row.get("close_azimuth", 125),
            slot.row.get("close_elevation", -25),
        )
        if slot.row["kind"] == "tennis":
            camera.lookat[:] = slot.origin + [1.8, 0, 0.7]
            camera.distance = 6.5
        renderer.update_scene(arena.data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        panel = Image.fromarray(renderer.render().copy()).resize((480, 300))
        y = top + 300 * j
        frame.paste(panel, (1440, y))
        draw.rectangle((1440, y, 1920, y + 27), fill=(15, 22, 30))
        title = f"{i + 1}. {slot.row['label']}"
        if clean and slot.episode > 1 and slot.episode_step <= 15:
            title += " / Reset"
        draw.text((1448, y + 3), title, font=small, fill="white")
        if slot.lost:
            draw.text((1450, y + 275), "TRACKING WARNING", font=small, fill="#ff8a80")
    if not clean:
        if arena.config.get("show_depth_panel", True):
            sensor = next(
                s
                for s in arena.slots
                if s.row["kind"] == ("rough" if int(arena.data.time // 4) % 2 else "vault")
            )
            if sensor.camera.last_image is not None:
                pixels = np.clip(sensor.camera.last_image * 255, 0, 255).astype(np.uint8)
                frame.paste(
                    Image.fromarray(pixels).resize((192, 108), Image.Resampling.NEAREST), (8, 968)
                )
                draw.text((214, 972), "LIVE ROBOT DEPTH INPUT", font=small, fill="white")
                draw.text((214, 999), sensor.row["label"], font=small, fill="#bdd0df")
                draw.text(
                    (214, 1026), "Frozen encoder / actual Actor input", font=tiny, fill="#bdd0df"
                )
        for i, s in enumerate(arena.slots):
            x = (650 if arena.config.get("show_depth_panel", True) else 25) + (i % 3) * (
                420 if arena.config.get("show_depth_panel", True) else 625
            )
            y = 970 + (i // 3) * 34
            color = "#ff8a80" if s.lost else "#b8e4c5"
            draw.text((x, y), f"{i + 1}. {s.row['label']}  ep{s.episode}", font=tiny, fill=color)
    return np.asarray(frame)
