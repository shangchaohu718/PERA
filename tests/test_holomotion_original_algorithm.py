import torch

from unified_motion.holomotion_original_algorithm import (
    ACTOR_CURRENT_TERM_WIDTHS,
    ACTOR_FUTURE_TERM_WIDTHS,
    CRITIC_CURRENT_TERM_WIDTHS,
    CRITIC_FUTURE_TERM_WIDTHS,
    flat_observations_to_tensordict,
)


def test_flat_observations_round_trip_original_tensordict_order():
    actor = torch.arange(2 * 506, dtype=torch.float32).reshape(2, 506)
    critic = torch.arange(2 * 843, dtype=torch.float32).reshape(2, 843)

    data = flat_observations_to_tensordict(actor, critic)

    actor_parts = [data[f"unified/{name}"] for name, _ in ACTOR_CURRENT_TERM_WIDTHS]
    actor_parts.extend(data[f"unified/{name}"] for name, _ in ACTOR_FUTURE_TERM_WIDTHS)
    rebuilt_actor = torch.cat(
        [
            *actor_parts[: len(ACTOR_CURRENT_TERM_WIDTHS)],
            *(
                value.reshape(2, -1)
                for value in actor_parts[len(ACTOR_CURRENT_TERM_WIDTHS) :]
            ),
        ],
        -1,
    )
    critic_parts = [data[f"unified/{name}"] for name, _ in CRITIC_CURRENT_TERM_WIDTHS]
    critic_parts.extend(
        data[f"unified/{name}"] for name, _ in CRITIC_FUTURE_TERM_WIDTHS
    )
    rebuilt_critic = torch.cat(
        [
            *critic_parts[: len(CRITIC_CURRENT_TERM_WIDTHS)],
            *(
                value.reshape(2, -1)
                for value in critic_parts[len(CRITIC_CURRENT_TERM_WIDTHS) :]
            ),
        ],
        -1,
    )
    assert torch.equal(rebuilt_actor, actor)
    assert torch.equal(rebuilt_critic, critic)


def test_sequence_observations_preserve_batch_and_time_dimensions():
    actor = torch.arange(2 * 3 * 506, dtype=torch.float32).reshape(2, 3, 506)
    critic = torch.arange(2 * 3 * 843, dtype=torch.float32).reshape(2, 3, 843)

    data = flat_observations_to_tensordict(actor, critic)

    assert tuple(data.batch_size) == (2, 3)
    assert data["unified/actor_dof_pos"].shape == (2, 3, 22)
    assert data["unified/actor_ref_dof_pos_fut"].shape == (2, 3, 10, 22)
    assert data["unified/critic_ref_dof_pos_fut"].shape == (2, 3, 10, 22)
