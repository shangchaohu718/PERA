"""Keep an explicitly authorized imperfect teacher distinct from acceptance."""

import hashlib
import json


def teacher_training_decision(gate, *, task, checkpoint_sha256, authorization=None):
    """Validate provenance and permit only the specific, numerically valid waiver."""
    if (
        gate.get("status") != "completed"
        or gate.get("arguments", {}).get("task") != task
        or gate.get("teacher_checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("Teacher report is incomplete or for another task/checkpoint")
    evaluation = gate["evaluation"]
    accepted = evaluation["acceptance_passed"] is True
    result = {
        "teacher_acceptance_passed": accepted,
        "training_allowed": accepted,
        "authorization_used": False,
        "teacher_successes": evaluation.get("successes"),
        "teacher_episodes": evaluation.get("num_episodes"),
    }
    if accepted:
        return result
    if authorization is None:
        raise ValueError("Failed teacher requires explicit user authorization")
    if (
        task != "stairs_up"
        or authorization.get("task") != task
        or authorization.get("user_authorized") is not True
        or authorization.get("user_response") != "也蒸馏，保留教师失败率"
        or authorization.get("allow_physical_failures") is not True
        or authorization.get("require_demonstrated_success") is not True
        or authorization.get("minimum_episodes") != 8
        or authorization.get("teacher_checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("Authorization does not cover this teacher/experiment")
    episodes = evaluation.get("episodes", [])
    successes = sum(e.get("success") is True for e in episodes)
    if (
        len(episodes) < authorization["minimum_episodes"]
        or successes == 0
        or any(e.get("physics_valid") is not True for e in episodes)
        or successes != evaluation.get("successes")
        or len(episodes) != evaluation.get("num_episodes")
    ):
        raise ValueError(
            "Teacher has no demonstrated success, invalid physics or inconsistent counts"
        )
    result.update(
        training_allowed=True,
        authorization_used=True,
        user_response=authorization["user_response"],
        teacher_successes=successes,
        teacher_episodes=len(episodes),
        teacher_success_rate=successes / len(episodes),
        authorization_sha256=hashlib.sha256(
            json.dumps(authorization, sort_keys=True).encode()
        ).hexdigest(),
    )
    return result
