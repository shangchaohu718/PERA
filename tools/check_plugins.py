"""R21插口检查：逐个运行六种策略×两种物理编码。

这是解析诊断，不是K1/G1仿真；不宣称任何动作成功或轻量化性能优势。
输入：--cases可以只选部分配置（例如gru,gnn_gru）；默认检查全部12种。
输出：runs中的诊断权重、每次训练/冻结日志，以及reports/current/plugin_matrix.json。
所有模型都使用相同的小预算；参数数量与结果真实记录，不手填通过数字。
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from unified_motion.experiment import load_config, run_training, run_evaluation, describe_models


def main():
    """解析选择的配置，执行训练与冻结评测；失败直接终止，不能把失败当跳过。

    使用者应在项目根目录运行，YAML的output保持相对路径，方便独立解压复现。
    成功率可以为0：本工具检查连接和冻结条件，不把成功率改成通过标记。
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',default='mlp,transformer,moe,gru,lstm,lru,gnn_mlp,gnn_transformer,gnn_moe,gnn_gru,gnn_lstm,gnn_lru')
    args=parser.parse_args()
    names=args.cases.split(',')
    allowed={f'{prefix}{kind}' for prefix in ('','gnn_') for kind in ('mlp','dense','transformer','moe','gru','lstm','lru')}
    if len(names)!=len(set(names)) or set(names)-allowed:
        parser.error('cases包含重复或未知配置')
    torch.set_num_threads(2)
    directory=ROOT/'reports/current';directory.mkdir(parents=True,exist_ok=True)
    report=[]
    for name in names:
        filename='diagnostic.yaml' if name=='mlp' else f'diagnostic_{name}.yaml'
        config=load_config(ROOT/'configs'/filename)
        with (directory/f'smoke_{name}.txt').open('w',encoding='utf-8') as stream:
            with redirect_stdout(stream):
                checkpoint=run_training(config)
                result=run_evaluation(config,checkpoint)
        if not result['completed'] or not result['weights_unchanged']:
            raise RuntimeError(f'{name}未通过完整回合/参数冻结检查')
        rows=[json.loads(line) for line in (Path(config['output'])/'metrics.jsonl').read_text().splitlines()]
        current=rows[-config['run']['updates']:]
        if not any(row['dynamics'] is not None for row in current):
            raise RuntimeError(f'{name}没有执行独立动力学更新')
        old_error=max(row['ppo']['old_policy_error'] for row in current)
        item=dict(case=name,parameters=describe_models(config),training_updates=len(current),
                  old_policy_max_error=old_error,evaluation=result,
                  checkpoint=str(checkpoint),log=f'smoke_{name}.txt')
        report.append(item)
        (directory/'plugin_matrix.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{name}: train={len(current)} old_policy_error={old_error:.3g} frozen={result["weights_unchanged"]} success_rate={result["success_rate"]}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
