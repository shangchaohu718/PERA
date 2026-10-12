"""Unified K1 training entry; the historical HoloMotion command remains supported."""
if __package__:
    from .train_holomotion_k1_mjlab import main
else:
    from train_holomotion_k1_mjlab import main

if __name__ == "__main__":
    main()
