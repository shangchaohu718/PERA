import numpy as np

from tools.convert_holomotion_eval_reference import _pose_gradient


def test_pose_gradient_matches_holomotion_boundary_and_central_rules():
    values = np.array([[0.0], [1.0], [4.0], [9.0]], dtype=np.float32)

    velocity = _pose_gradient(values, fps=2.0)

    np.testing.assert_array_equal(
        velocity,
        np.array([[2.0], [4.0], [8.0], [10.0]], dtype=np.float32),
    )


def test_pose_gradient_single_frame_is_zero():
    velocity = _pose_gradient(np.array([[3.0, 4.0]], dtype=np.float32), fps=50.0)
    np.testing.assert_array_equal(velocity, np.zeros((1, 2), dtype=np.float32))
