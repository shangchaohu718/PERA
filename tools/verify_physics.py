"""物理验收命令：源码及原理在 unified_motion.envs.physics_validation。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_motion.envs.physics_validation import main

if __name__ == "__main__":
    raise SystemExit(main())
