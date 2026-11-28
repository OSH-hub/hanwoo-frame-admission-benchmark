from pathlib import Path
import argparse,io
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def render(metrics,output):
    d=pd.read_csv(metrics)
    assert len(d)==36 and set(d.policy)=={'latest','drop_newest','periodic'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    styles=[('latest','Latest','#0072B2','o',11,'none'),
            ('drop_newest','Drop newest','#D55E00','s',7.5,'none'),
            ('periodic','Periodic','#009E73','^',4.5,'#009E73')]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,['p99_ms','mean_aoi_ms'],['Delivered latency p99 (ms)','Mean information age (ms)']):
        for policy,label,color,marker,size,fill in styles:
            r=d[d.policy==policy]
            g=r.groupby('admission_fraction').agg(drop=('drop_pct','mean'),value=(metric,'mean'))
            ax.plot(g['drop'],g['value'],color=color,lw=1.6,marker=marker,markersize=size,
                    markerfacecolor=fill,markeredgecolor=color,markeredgewidth=1.2,label=label)
            ax.scatter(r.drop_pct,r[metric],s=16,alpha=.35,color=color,marker=marker,linewidths=.6)
        ax.set_yscale('log');ax.set_xlabel('Actual discarded arrivals (%)');ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend(frameon=False);fig.suptitle('June source: fixed admission-fraction sensitivity (three runs each)',fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg']:
        dst=output.with_suffix('.'+ext)
        if dst.exists():dst.read_bytes()
        fig.savefig(dst,bbox_inches='tight',facecolor='white')
    memory=io.BytesIO();fig.savefig(memory,format='png',dpi=600,bbox_inches='tight',facecolor='white')
    dst=output.with_suffix('.png')
    if dst.exists():dst.read_bytes()
    with Image.open(memory) as im:im.convert('RGB').save(dst,dpi=(600,600))
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metrics',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.metrics,a.output)
