"""统一测评：用独立pytest进程运行全部测试，兼容unittest类和函数式测试。

输入：--output报告目录；--require-sim-imports要求真实仿真依赖可导入。
输出：cpu_regression.txt、evaluation.json和退出码。
0=执行的测试通过；1=失败/收集错误；2=额外真实依赖检查阻塞。
代码回归不代表真实物理模型编译、控制技能或CUDA吞吐得到验证。
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """启动完整测试进程，并把实际结果与依赖状态写入文件。

    先运行测试，再读取PyTorch版本，避免在父进程提前建立数值线程池。
    若子进程未生成报告，按失败处理，绝不沿用上次报告数字。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/current')
    parser.add_argument('--require-sim-imports', action='store_true')
    args = parser.parse_args()
    destination = args.output.resolve(); destination.mkdir(parents=True, exist_ok=True)
    raw = destination/'pytest_events.json'
    raw.unlink(missing_ok=True)
    env = os.environ.copy()
    env['UNIFIED_MOTION_PYTEST_REPORT'] = str(raw)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(key, '2')
    start = time.perf_counter()
    with (destination/'cpu_regression.txt').open('w', encoding='utf-8') as stream:
        completed = subprocess.run([sys.executable, '-m', 'pytest', '-q',
            '-p', 'tools._pytest_report', str(ROOT/'tests')], cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT)
    report = (json.loads(raw.read_text()) if raw.exists() else
              dict(total=0, passed=0, failed=1, skipped=0,
                   collection_errors=['pytest未生成本次报告'], tests=[]))
    import torch
    import numpy as np
    optional = {}
    for name in ('mujoco', 'mjlab', 'warp'):
        try:
            module=importlib.import_module(name)
            optional[name]=dict(importable=True, version=str(getattr(module,'__version__','not exposed')))
        except Exception as exc:
            optional[name]=dict(importable=False, reason=f'{type(exc).__name__}: {exc}')
    good = completed.returncode == 0 and raw.exists() and not report.get('collection_errors')
    report.update(project='0.21.0', evaluated_at=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), torch=str(torch.__version__), numpy=str(np.__version__),
        cuda_available=torch.cuda.is_available(), runner='pytest; includes unittest and pytest functions',
        process_returncode=completed.returncode, elapsed_seconds=time.perf_counter()-start,
        optional_dependencies=optional, core_suite_status='PASS' if good else 'FAIL',
        actual_mjlab_training='NOT_RUN: tensor/analytic/mock tests are not a robot contact solver',
        scope_note='通过代码测试不等于真实机器人仿真、GPU性能、零样本能力通过。')
    (destination/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='tests'},ensure_ascii=False,indent=2),flush=True)
    if not good:
        return 1
    return 2 if args.require_sim_imports and any(not v['importable'] for v in optional.values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
