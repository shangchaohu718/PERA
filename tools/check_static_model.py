"""旧静态编译命令的兼容入口。实际工作统一交给物理验收模块。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_motion.envs.physics_validation import main

if __name__ == "__main__":
    raise SystemExit(main(["--phase", "compile", *sys.argv[1:]]))
