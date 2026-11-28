from pathlib import Path
import json, re, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
SRC=BASE/'reproducibility/extended_live'
TABLE=BASE/'tables'; FIG=BASE/'figures'
POLICIES=['fifo','latest','drop_newest','periodic']
COLORS=['#4C566A','#0072B2','#D55E00','#009E73']

def age_stats(d,left,right):
    f=d.finish_ms.to_numpy();a=d.arrival_ms.to_numpy()
    k=np.searchsorted(f,left,side='right')-1
    assert k>=0 and right>left
    now=left;arrival=a[k];area=0.;maximum=now-arrival
    for j in range(k+1,len(d)):
        end=f[j]
        if end>right:break
        area+=((end-arrival)**2-(now-arrival)**2)/2
        maximum=max(maximum,end-arrival);now=end;arrival=a[j]
    area+=((right-arrival)**2-(now-arrival)**2)/2
    maximum=max(maximum,right-arrival)
    return area/(right-left),maximum

def retained_window_fraction(ids,n,w):
    keep=np.zeros(n,dtype=np.int64);keep[np.asarray(ids,dtype=int)]=1
    cumulative=np.concatenate(([0],np.cumsum(keep)))
    hits=cumulative[w:]-cumulative[:-w]
    return float(np.mean(hits>0))

def main():
    assert (SRC/'COMPLETE').exists(), 'Wait until every prespecified arm is complete.'
    meta=json.loads((SRC/'metadata.json').read_text())
    summaries=json.loads((SRC/'summaries.json').read_text())
    assert len(summaries)==25
    delivered={s['label']:pd.read_csv(SRC/f"{s['label']}_delivered.csv",keep_default_na=False) for s in summaries}
    windows={}
    for source in range(3):
        for fps in [30,80]:
            selected=[s for s in summaries if s['source']==source and s['fps']==fps and s['duration_s']==60]
            windows[(source,fps)]=(max(delivered[s['label']].finish_ms.iloc[0] for s in selected),(int(fps*60)-1)*1000/fps)
    metrics=[];coverage=[];telemetry=[]
    for s in summaries:
        label=s['label'];d=delivered[label];a=pd.read_csv(SRC/f'{label}_arrivals.csv')
        dropped_file=SRC/f'{label}_dropped.csv'
        drops=pd.read_csv(dropped_file).frame.tolist() if dropped_file.exists() else []
        n=s['n'];ids=d.frame.to_numpy(dtype=int)
        assert len(ids)+len(drops)==len(a)==n
        assert sorted(list(ids)+drops)==list(range(n))
        assert not d.error.astype(bool).any()
        assert np.allclose(d.finish_ms-d.arrival_ms,d.latency_ms,atol=1e-8)
        assert np.allclose(d.wait_ms+d.service_ms,d.latency_ms,atol=1e-8)
        assert np.allclose(d.arrival_ms,a.set_index('frame').loc[ids,'arrival_ms'],atol=1e-8)
        if s['duration_s']==60:left,right=windows[(s['source'],s['fps'])]
        else:left,right=float(d.finish_ms.iloc[0]),(n-1)*1000/s['fps']
        age,age_max=age_stats(d,left,right)
        gaps=np.diff(d.finish_ms)
        m=dict(s,mean_aoi_ms=age,max_aoi_ms=age_max,common_window_start_ms=left,common_window_end_ms=right,
            completion_gap_p99_ms=float(np.percentile(gaps,99)),max_replay_arrival_index_step=int(np.diff(ids).max()),
            scheduling_lateness_p99_ms=float(np.percentile(a.arrival_ms-a.scheduled_ms,99)),
            median_detections=float(d.n_detections.median()),finish_ms=float(d.finish_ms.iloc[-1]))
        for key,expected in [('p99_ms',np.percentile(d.latency_ms,99)),('drop_pct',100*len(drops)/n),('miss_pct',100*np.mean(d.latency_ms>33))]:
            assert abs(s[key]-expected)<1e-8
        metrics.append(m)
        k=len(ids);equal_count=np.floor((np.arange(k)+.5)*n/k).astype(int)
        assert len(np.unique(equal_count))==k
        for w in [1,2,4,8,16]:
            coverage.append(dict(label=label,source=s['source'],fps=s['fps'],policy=s['policy'],duration_s=s['duration_s'],
                scheduled_frames=n,retained_frames=k,window_frames=w,nominal_window_ms=w*1000/s['fps'],
                observed_retained_window_fraction=retained_window_fraction(ids,n,w),
                equal_count_evenly_spaced_fraction=retained_window_fraction(equal_count,n,w)))
        for t in json.loads((SRC/f'{label}_telemetry.json').read_text()):
            fields=t['values'].split(',')
            if len(fields)==5:
                telemetry.append(dict(label=label,elapsed_s=t['elapsed_s'],gpu_timestamp=fields[0].strip(),
                    temperature_c=float(fields[1]),power_w=float(fields[2]),utilization_pct=float(fields[3]),memory_mib=float(fields[4])))
    df=pd.DataFrame(metrics);cv=pd.DataFrame(coverage);tm=pd.DataFrame(telemetry)
    for p in [TABLE,FIG]:p.mkdir(exist_ok=True)
    df.to_csv(TABLE/'gpu1_extension_all_arms.csv',index=False)
    df[df.duration_s==600].to_csv(TABLE/'gpu1_ten_minute_arm.csv',index=False)
    cv.to_csv(TABLE/'gpu1_retained_window_diagnostic.csv',index=False)
    tm.to_csv(TABLE/'gpu1_extension_gpu_telemetry.csv',index=False)
    short=df[df.duration_s==60]
    paired=[]
    for source in range(3):
        sub=short[(short.source==source)&(short.fps==80)].set_index('policy')
        for control in ['fifo','drop_newest','periodic']:
            paired.append(dict(source=source,comparison=f'latest minus {control}',
                p99_difference_ms=sub.loc['latest','p99_ms']-sub.loc[control,'p99_ms'],
                aoi_difference_ms=sub.loc['latest','mean_aoi_ms']-sub.loc[control,'mean_aoi_ms'],
                drop_difference_percentage_points=sub.loc['latest','drop_pct']-sub.loc[control,'drop_pct']))
    pd.DataFrame(paired).to_csv(TABLE/'gpu1_extension_paired_differences.csv',index=False)
    agg=short.groupby(['fps','policy']).agg(p99_min=('p99_ms','min'),p99_max=('p99_ms','max'),
        aoi_min=('mean_aoi_ms','min'),aoi_max=('mean_aoi_ms','max'),drop_min=('drop_pct','min'),drop_max=('drop_pct','max')).reset_index()
    agg.to_csv(TABLE/'gpu1_extension_summary.csv',index=False)
    plt.rcParams.update({'font.size':12.5,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10,4.6))
    sub=short[short.fps==80]
    for ax,key,title in zip(axes,['p99_ms','mean_aoi_ms','drop_pct'],['Delivered latency p99\n(ms)','Time-average age\n(ms)','Discarded arrivals\n(%)']):
        for j,(policy,color) in enumerate(zip(POLICIES,COLORS)):
            vals=sub[sub.policy==policy].sort_values('source')[key].values
            ax.scatter(np.arange(3)+(j-1.5)*.12,vals,color=color,label=['FIFO','Latest','Drop newest','Periodic'][j],s=34,marker=['o','s','^','D'][j])
        if key!='drop_pct':ax.set_yscale('log')
        ax.set_xticks(range(3),['Public\ndairy','NVR\nJune','NVR\nNovember']);ax.set_title(title,fontsize=12.5);ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=12.5,frameon=False)
    fig.suptitle('One-minute GPU tests at 80 artificial arrivals/s',fontsize=14)
    fig.tight_layout(rect=[0,.12,1,.95])
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'figure4_extended.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)
    label='s1_80_latest_burst600';d=delivered[label]
    bins=[]
    for left in range(1000,599000,1000):
        age,_=age_stats(d,left,left+1000)
        block=d[(d.finish_ms>=left)&(d.finish_ms<left+1000)]
        bins.append(dict(second=left/1000,mean_aoi_ms=age,p99_ms=float(np.percentile(block.latency_ms,99)) if len(block) else np.nan))
    bd=pd.DataFrame(bins);bd.to_csv(TABLE/'gpu1_ten_minute_one_second_bins.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(9,5.5),sharex=True)
    axes[0].plot(bd.second,bd.mean_aoi_ms,label='One-second mean age',lw=1)
    axes[0].plot(bd.second,bd.p99_ms,label='One-second delivered p99',lw=.8,alpha=.75)
    axes[0].set_ylabel('Milliseconds')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=11.5,frameon=False)
    t=tm[tm.label==label];axes[1].plot(t.elapsed_s,t.power_w,color='#D55E00',lw=1)
    axes[1].set_ylabel('GPU power (W)');axes[1].set_xlabel('Elapsed replay time (s)')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Ten-minute latest-pending run with alternating matrix load');fig.tight_layout(rect=[0,.09,1,1])
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'figure5_burst.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)
    summary=dict(arm_count=len(df),scheduled_arrivals=int(df.n.sum()),delivered=int(df.delivered.sum()),
        errors=int(df.errors.sum()),source_count=len(meta['sources']),decoded_frames=[s['decoded_frames'] for s in meta['sources']],
        aggregate=agg.to_dict('records'),long_run=df[df.duration_s==600].iloc[0].to_dict(),
        paired=paired,scope='Heterogeneous source blocks; descriptive ranges, no device-independent CI; hypothetical window retention only.')
    (SRC/'recomputed_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
