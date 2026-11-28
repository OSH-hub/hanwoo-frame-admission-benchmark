from pathlib import Path
import json, hashlib, shutil, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
SRC=OUT.parents[1]/'system/research/p2_digital_twin'
if not SRC.exists():SRC=OUT/'reproducibility/historical'
for folder in ['tables','figures','reproducibility/historical']:(OUT/folder).mkdir(parents=True,exist_ok=True)
manifest=[]
def read_csv(path):
    rel=path.relative_to(SRC); dst=OUT/'reproducibility/historical'/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    if path.resolve()!=dst.resolve():shutil.copy2(path,dst)
    manifest.append(dict(file=str(rel),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    return pd.read_csv(path)
def read_json(path):
    rel=path.relative_to(SRC);dst=OUT/'reproducibility/historical'/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    if path.resolve()!=dst.resolve():shutil.copy2(path,dst)
    manifest.append(dict(file=str(rel),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    return json.loads(path.read_text())
def metrics(df,n):
    x=df.e2e_ms.to_numpy()
    return dict(delivered=len(df),dropped=n-len(df),drop_pct=100*(n-len(df))/n,
        p99_ms=np.percentile(x,99),max_ms=x.max(),miss_pct=100*np.mean(x>33),
        error_rows=int(df.inference_error.astype(str).str.lower().eq('true').sum()) if 'inference_error' in df else 0)

lat=read_csv(SRC/'results/latency_per_frame.csv')
rows=[]
for col in ['preprocess_ms','inference_ms','postprocess_ms','twin_sync_ms','e2e_ms']:
    x=lat[col].to_numpy();rows.append(dict(stage=col,n=len(x),mean_ms=x.mean(),sd_ms=x.std(ddof=1),p95_ms=np.percentile(x,95),p99_ms=np.percentile(x,99),max_ms=x.max()))
pd.DataFrame(rows).to_csv(OUT/'tables/historical_stage_timing.csv',index=False)
hist=[];checks=[]
for run in range(1,6):
    meta=read_json(SRC/f'results_fixedC/run_{run}/results.json')
    for arm in ['baseline','dats']:
        frame=read_csv(SRC/f'results_fixedC/run_{run}/per_frame_{arm}.csv')
        row=dict(run=run,arm=arm,C=meta['calibration']['C_achieved'],rho=meta['calibration']['rho_achieved'],**metrics(frame,300))
        hist.append(row)
        old=meta['live_results'][arm]
        checks.append(dict(run=run,arm=arm,csv_delivered=len(frame),json_delivered=old['delivered'],
            csv_p99=row['p99_ms'],json_p99=old.get('e2e_p99_ms'),
            delivered_match=len(frame)==old['delivered']))
pd.DataFrame(hist).to_csv(OUT/'tables/historical_stress_runs.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'tables/historical_csv_json_consistency.csv',index=False)
moderate=[]
for arm in ['baseline','dats']:
    f=read_csv(SRC/f'results/live_ab_per_frame_{arm}.csv')
    moderate.append(dict(arm=arm,**metrics(f,600)))
pd.DataFrame(moderate).to_csv(OUT/'tables/historical_moderate_load.csv',index=False)
power=read_json(SRC/'results/batch_power_results.json')
pd.DataFrame([dict(mode=m,frames=power[m]['frames'],wall_s=power[m]['wall_s'],fps=power[m]['fps'],power_w=power[m]['power']['power_mean_w'],energy_j=power['energy_per_frame_j'][m]) for m in ['fp32','fp16']]).to_csv(OUT/'tables/energy_short_runs.csv',index=False)

def simulate(svc,policy,T=1000/30):
    queue=[];busy=None;free=float('inf');rows=[];drop=[];i=0;n=len(svc)
    while i<n or busy is not None or queue:
        a=i*T if i<n else float('inf')
        if busy is not None and free<=a:
            idx,arrival,start=busy;rows.append((idx,arrival,start,free));busy=None
            if queue:
                idx,arrival=queue.pop(0);busy=(idx,arrival,free);free=free+svc[idx]
            else:free=float('inf')
            continue
        if i<n:
            if busy is None:busy=(i,a,a);free=a+svc[i]
            else:
                queue.append((i,a))
                if policy=='latest':
                    while len(queue)>1:drop.append(queue.pop(0)[0])
            i+=1
        else:raise RuntimeError('invalid event state')
    r=np.array(rows);assert len(r)+len(drop)==n
    return r,drop

base=lat[['preprocess_ms','inference_ms','postprocess_ms','twin_sync_ms']].sum(axis=1).to_numpy()
rng=np.random.default_rng(42);svc=rng.choice(base,1500);svc[500:1000]*=4
sim=[]
for pol in ['fifo','latest']:
    r,drop=simulate(svc,pol);x=r[:,3]-r[:,1]
    sim.append(dict(policy=pol,delivered=len(r),dropped=len(drop),drop_pct=100*len(drop)/1500,
        miss_pct=100*np.mean(x>33),p99_ms=np.percentile(x,99),max_ms=x.max()))
    pd.DataFrame(r,columns=['frame','arrival_ms','start_ms','finish_ms']).to_csv(OUT/f'reproducibility/sim_{pol}.csv',index=False)
pd.DataFrame(sim).to_csv(OUT/'tables/chronological_simulation.csv',index=False)
r,d=simulate(np.full(40,40.),'latest');counter=float(max(r[:,3]-r[:,1]))
assert counter>40+max(40-1000/30,0)
for seed in range(30):
    s=np.random.default_rng(seed).uniform(1,150,300);r,d=simulate(s,'latest')
    assert max(r[:,3]-r[:,1])<=2*max(s)+1e-8

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,ax=plt.subplots(figsize=(6.4,3.8));ax.hist(lat.e2e_ms,bins=20,color='#285780',edgecolor='white');ax.set(xlabel='Frame processing latency (ms)',ylabel='Frames');fig.tight_layout();fig.savefig(OUT/'figures/figure1_latency.png',dpi=300);fig.savefig(OUT/'figures/figure1_latency.pdf');plt.close(fig)
h=pd.DataFrame(hist);fig,ax=plt.subplots(figsize=(6.4,4));xs=np.arange(5)
for arm,offset,color,label in [('baseline',-.17,'#9b3535','FIFO'),('dats',.17,'#285780','Historical DATS')]:
    ax.bar(xs+offset,h[h.arm==arm].p99_ms,width=.32,color=color,label=label)
ax.set_yscale('log');ax.set_xticks(xs,[str(x) for x in range(1,6)]);ax.set(xlabel='Historical stress run',ylabel='Delivered latency p99 (ms)')
handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(OUT/'figures/figure2_historical.png',dpi=300);fig.savefig(OUT/'figures/figure2_historical.pdf');plt.close(fig)
(OUT/'reproducibility/source_manifest.json').write_text(json.dumps(manifest,indent=2))
summary=dict(historical=hist,moderate=moderate,stages=rows,simulation=sim,
    withdrawn_bound_counterexample_ms=counter,withdrawn_formula_ms=40+max(40-1000/30,0),
    source_mismatches=[x for x in checks if not x['delivered_match'] or (x['json_p99'] is not None and abs(x['csv_p99']-x['json_p99'])>.01)])
(OUT/'reproducibility/reanalysis.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
