"""A reference boundary must reset only its robot, clock, action and history."""

from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
import torch
from pera.showcase import SharedWorld


def test_reset_isolates_physical_state_and_episode_history(tmp_path):
    bodies = []
    actuators = []
    for i in range(2):
        chain = "".join(
            f'<body><joint name="j{i}_{j}" type="hinge"/><geom size=".01" mass=".01"/>'
            for j in range(29)
        )
        chain += "</body>" * 29
        actuators.extend(f'<position joint="j{i}_{j}" kp="10"/>' for j in range(29))
        bodies.append(
            f'<body pos="{i * 3} 0 1"><freejoint/><geom size=".05" mass="1"/>{chain}</body>'
        )
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        + "".join(bodies)
        + "</worldbody><actuator>"
        + "".join(actuators)
        + "</actuator></mujoco>"
    )
    w = SharedWorld.__new__(SharedWorld)
    w.model, w.data = model, mujoco.MjData(model)
    w.data.time = 12.0
    w.root_q, w.root_v = [0, 36], [0, 35]
    w.q_ids = np.array([np.arange(7, 36), np.arange(43, 72)])
    w.v_ids = np.array([np.arange(6, 35), np.arange(41, 70)])
    w.u_ids = np.arange(58).reshape(2, 29)
    raw = tmp_path / "motion.npz"
    np.savez(
        raw,
        body_lin_vel_w=np.ones((1, 1, 3)) * 0.2,
        body_ang_vel_w=np.zeros((1, 1, 3)),
        joint_vel=np.ones((1, 29)) * 0.3,
    )
    w.tasks = [dict(id=f"task{i}", motion=str(raw)) for i in range(2)]
    p = torch.tensor([[0.0, 0, 1], [0, 0, 1]])
    R = torch.eye(3).repeat(2, 1, 1)
    q = torch.ones(2, 29) * 0.1
    w.motion = SimpleNamespace(
        clip_start=torch.tensor([0, 1]), ref_state=lambda f: (p, R, p * 0, p * 0, q, q * 0)
    )
    w.origins = torch.tensor([[0.0, 0, 0], [3, 0, 0]])
    w.control_step = 58
    w.episode_step = torch.tensor([58, 58])
    w.episode_id = np.array([1, 1])
    w.tracking_lost = np.array([True, True])
    w.reset_events = []
    w.last_action = torch.ones(2, 29)
    w.initial_previous_action = torch.zeros(29)
    pending = torch.zeros(2, dtype=torch.bool)
    w.controller = SimpleNamespace(reset=lambda ids: pending.__setitem__(ids, True))
    w.data.qvel[:] = 0.8
    w.data.qacc_warmstart[:] = 0.9
    before_q, before_v = w.data.qpos.copy(), w.data.qvel.copy()
    w.reset([1])
    np.testing.assert_array_equal(w.data.qpos[:36], before_q[:36])
    np.testing.assert_array_equal(w.data.qvel[:35], before_v[:35])
    np.testing.assert_allclose(w.data.qpos[36:39], [3, 0, 1])
    np.testing.assert_allclose(w.data.qpos[43:], 0.1)
    np.testing.assert_allclose(w.data.qvel[41:], 0.3)
    np.testing.assert_allclose(w.data.ctrl[:29], 0)
    np.testing.assert_allclose(w.data.ctrl[29:], 0.1)
    assert w.data.time == 12 and w.control_step == 58
    assert w.episode_step.tolist() == [58, 0]
    assert w.episode_id.tolist() == [1, 2]
    assert pending.tolist() == [False, True]
    assert w.tracking_lost.tolist() == [True, False]
    assert w.last_action[0].eq(1).all() and w.last_action[1].eq(0).all()
    assert len(w.reset_events) == 1 and w.reset_events[0]["tracking_lost"]
    with pytest.raises(ValueError):
        w.reset([-1])


def test_cpu_mlp_does_not_enter_cuda_autocast(monkeypatch):
    from blinkmimic import networks

    monkeypatch.setattr(networks, "AMP_DTYPE", torch.bfloat16)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU inference entered CUDA autocast")

    monkeypatch.setattr(torch, "autocast", forbidden)
    net = torch.nn.Linear(3, 2)
    x = torch.randn(2, 3)
    torch.testing.assert_close(networks.run_mlp(net, x), net(x), atol=0, rtol=0)


def test_tennis_mirror_preserves_native_step_sensor_timing():
    from pera.interaction_showcase import Mirror

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep=".0025" gravity="0 0 0"/>
      <worldbody><body name="robot"><freejoint/><geom size=".1" mass="1"/>
      <site name="imu"/></body></worldbody>
      <sensor><gyro site="imu" name="gyro"/></sensor></mujoco>""")
    data, native, mirror_data = (mujoco.MjData(model) for _ in range(3))
    for d in (data, native, mirror_data):
        d.xfrc_applied[1, 5] = 0.1
        mujoco.mj_forward(model, d)
    mirror = Mirror.__new__(Mirror)
    mirror.slot = SimpleNamespace(model=model, data=mirror_data, origin=np.zeros(3))
    mirror.qmap, mirror.vmap, mirror.umap = np.arange(7), np.arange(6), np.arange(0)
    mirror.free = [0]
    arena = SimpleNamespace(data=data)
    for _ in range(8):
        mirror.pull(arena)
        mujoco.mj_step(model, native)
        mujoco.mj_step(model, data)
        mirror.pull(arena, forward=False)
        np.testing.assert_array_equal(mirror_data.qpos, native.qpos)
        np.testing.assert_array_equal(mirror_data.qvel, native.qvel)
        np.testing.assert_array_equal(mirror_data.sensordata, native.sensordata)
    previous_sensor = mirror_data.sensordata.copy()
    mujoco.mj_forward(model, mirror_data)
    assert np.max(abs(previous_sensor - mirror_data.sensordata)) > 0.01


def test_native_50hz_reference_preserves_terminal_frame():
    from blinkmimic.motion import MotionLib

    p = np.arange(59 * 3, dtype=np.float32).reshape(59, 3) / 100
    q = np.arange(59 * 29, dtype=np.float32).reshape(59, 29) / 1000
    quat = np.zeros((59, 4), np.float32)
    quat[:, 0] = 1
    kin = SimpleNamespace(
        forward=lambda p, R, v, w, q, qd: (p[:, None], R[:, None], v[:, None], w[:, None])
    )
    motion = MotionLib(["native"], kin, "cpu", lambda _: (p, quat, q, 50.0))
    assert int(motion.clip_end[0]) == 58 and motion.num_frames == 59
    torch.testing.assert_close(motion.root_p, torch.from_numpy(p), atol=0, rtol=0)
    torch.testing.assert_close(motion.q, torch.from_numpy(q), atol=0, rtol=0)


def test_depth_camera_continues_past_near_self_geometry():
    from pera.interaction_showcase import VaultCamera
    from torch.nn import functional as F

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="torso"><geom type="box" size=".005 .2 .2" pos=".02 0 0"/></body>
      <geom type="box" size=".005 2 2" pos="1 0 0"/>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = VaultCamera.__new__(VaultCamera)
    camera.device = "cpu"
    camera.offset_R = np.eye(3)
    camera.geometry = dict(
        offset_pos=np.zeros(3),
        starts=np.zeros((1296, 3)),
        directions=np.tile([1.0, 0, 0], (1296, 1)),
    )

    def process(x, cfg, ids):
        return F.interpolate(
            x.permute(0, 3, 1, 2) / 2, size=(18, 32), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

    camera.pipeline = [SimpleNamespace(func=process)]
    camera.encoder = lambda x: x.mean().expand(1, 32)
    context = camera.observe(model, data, model.body("torso").id)
    assert camera.last_image.shape == (18, 32)
    np.testing.assert_allclose(camera.last_image, 0.995 / 2, atol=1e-6)
    assert context.isfinite().all()


def test_compact_regions_preserve_contacts_and_isolate_neighbors():
    from pera.compact_showcase import compact_child_spec

    specs = []
    for i in range(9):
        spec = mujoco.MjSpec.from_string("""<mujoco><worldbody>
          <geom name="floor" type="plane" size="2 2 .1" contype="1" conaffinity="3"/>
          <body pos="0 0 .4"><freejoint/>
            <geom name="robot" size=".1" contype="2" conaffinity="1"/>
          </body></worldbody></mujoco>""")
        compact_child_spec(spec, {"kind": "motion"}, i)
        specs.append(spec)

    def contacts(a, b):
        return bool((a.contype & b.conaffinity) or (b.contype & a.conaffinity))

    for i, spec in enumerate(specs):
        assert contacts(spec.geom("floor"), spec.geom("robot"))
        for j, other in enumerate(specs):
            if i != j:
                for a in spec.geoms:
                    for b in other.geoms:
                        assert not contacts(a, b)
        spec.compile()


def test_compact_masks_preserve_sparse_native_box_contact_graph():
    from pera.compact_showcase import compact_child_spec

    masks = [(1, 1008), (16, 961), (32, 897), (64, 913), (128, 625), (256, 113), (512, 241)]
    spec = mujoco.MjSpec()
    for i, (a, b) in enumerate(masks):
        spec.worldbody.add_geom(name=f"g{i}", size=[0.1, 0, 0], contype=a, conaffinity=b)
    used = compact_child_spec(spec, {"kind": "box"}, 0, bit_offset=3)
    assert used == 7
    for i, (a, b) in enumerate(masks):
        for j, (c, d) in enumerate(masks):
            x, y = spec.geoms[i], spec.geoms[j]
            assert bool((a & d) or (c & b)) == bool(
                (x.contype & y.conaffinity) or (y.contype & x.conaffinity)
            )
