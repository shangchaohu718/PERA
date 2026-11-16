from pera.smoke import verify


def test_frozen_competitive_routing_and_replay_contracts():
    result = verify()
    assert result["public_unchanged"] and result["actual_adam_updates"] == 3
