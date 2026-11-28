from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
h=pd.read_csv(OUT/'tables/historical_stress_runs.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,ax=plt.subplots(figsize=(6.4,4));xs=np.arange(5)
for arm,offset,color,label in [('baseline',-.17,'#9b3535','FIFO'),('dats',.17,'#285780','Historical DATS')]:
    ax.bar(xs+offset,h[h.arm==arm].p99_ms,width=.32,color=color,label=label)
ax.set_yscale('log');ax.set_xticks(xs,[str(x) for x in range(1,6)])
ax.set(xlabel='Historical stress run',ylabel='Delivered latency p99 (ms)')
handles,labels=ax.get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
fig.tight_layout(rect=[0,.12,1,1])
for ext in ['png','pdf']:
    p=OUT/f'figures/figure2_historical.{ext}'
    if p.exists():previous=p.read_bytes()
    fig.savefig(p,dpi=300)
plt.close(fig)
