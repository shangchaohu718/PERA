"""Unified frozen K1 evaluation entry; Actor architecture comes from its bundle."""
if __package__:
    from .evaluate_physics_holomotion_k1_mjlab import main
else:
    from evaluate_physics_holomotion_k1_mjlab import main

if __name__ == "__main__":
    main()
