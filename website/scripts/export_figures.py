#!/usr/bin/env python3
"""Create shareable vector figures from the exact exported experiment snapshot."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WEB=Path(__file__).resolve().parents[1]
data=json.loads((WEB/'public/data/evidence.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#d2d2d7','axes.labelcolor':'#6e6e73','xtick.color':'#6e6e73','ytick.color':'#6e6e73','svg.fonttype':'none','savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
ax.plot([r['updates'] for r in data['curve']],[r['loss'] for r in data['curve']],color='#0071e3',linewidth=2,marker='o',markersize=3)
ax.set(xlabel='Cumulative Adam updates',ylabel='Mean action-supervision MSE',ylim=(0,.03),title='50 cm climbing expert · BC + online expert training')
ax.grid(axis='y',color='#ececef');ax.set_axisbelow(True)
fig.supxlabel('One mean per training round. Replay distribution changes; this is not held-out return.',fontsize=8,color='#6e6e73')
fig.savefig(WEB/'public/data/training-loss.svg');plt.close(fig)
rows=data['roundhouse']['clips'];y=np.arange(len(rows));fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
ax.barh(y-.16,[r['original']['mean_body_error_cm'] for r in rows],height=.3,color='#bfc0c6',label='Public policy')
ax.barh(y+.16,[r['expert_enabled']['mean_body_error_cm'] for r in rows],height=.3,color='#0071e3',label='+ roundhouse expert (enabled on every clip)')
ax.set_yticks(y,[r['name'].replace('_',' ') for r in rows]);ax.invert_yaxis();ax.set(xlabel='Mean body tracking error (cm) ↓',title='Roundhouse Expert · all seven recorded comparisons')
ax.grid(axis='x',color='#ececef');ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=8,loc='lower right')
fig.supxlabel('Seed 42 · nominal Isaac Lab physics · no fall terminations · final 10 reference frames omitted.',fontsize=8,color='#6e6e73')
fig.savefig(WEB/'public/data/body-error.svg');plt.close(fig)
print('Exported two source-backed vector figures.')
