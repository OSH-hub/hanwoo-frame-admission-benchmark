"""Recompute P2 run metrics from saved arrivals/completions; no source media/GPU."""
from pathlib import Path
import argparse, json, hashlib, math, io, platform
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

POLICIES=['fifo','latest','drop_newest','periodic']
COLORS={'fifo':'#59636b','latest':'#0072B2','drop_newest':'#D55E00','periodic':'#009E73'}
NAMES={'fifo':'FIFO','latest':'Latest','drop_newest':'Drop newest','periodic':'Periodic'}
SOURCE_NAMES=['Cows2021','June NVR','November NVR']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save_figure(fig,stem):
    for ext in ['pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),bbox_inches='tight',facecolor='white')
    memory=io.BytesIO();fig.savefig(memory,format='png',dpi=600,bbox_inches='tight',facecolor='white')
    with Image.open(memory) as rendered:
        rendered.convert('RGB').save(stem.with_suffix('.png'),dpi=(600,600))
def equal(a,b,msg,tol=1e-7):
    if not np.allclose(a,b,rtol=0,atol=tol,equal_nan=True):raise AssertionError(msg)

def independent_age(d,left,right):
    finish=d.finish_ms.to_numpy();arrival=d.arrival_ms.to_numpy()
    assert np.all(np.diff(finish)>0) and finish[0]<=left<right
    start=np.maximum(finish,left)
    end=np.minimum(np.r_[finish[1:],right],right)
    use=end>start
    widths=end[use]-start[use]
    equal(widths.sum(),right-left,'Incomplete AoI observation interval')
    low=start[use]-arrival[use];high=end[use]-arrival[use]
    area=math.fsum((widths*(low+high)/2).tolist())
    return area/(right-left),max(low.max(),high.max())

def interval(vals):
    x=np.asarray(vals,float);n=len(x);assert n==3
    mean=float(x.mean());sd=float(x.std(ddof=1));margin=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
    return dict(repetitions=n,mean=mean,SD=sd,minimum=float(x.min()),maximum=float(x.max()),
        t95_low=mean-margin,t95_high=mean+margin,df=n-1,
        inference='Conditional same-device run mean; not across-device/farm uncertainty')

def run(src,out,partial80=False):
    out.mkdir(parents=True,exist_ok=False)
    meta=load(src/'metadata.json');config=load(src/'config.json');design=load(src/'design.json');summaries=load(src/'summaries.json')
    equal(int(config['physical_gpu_index']),2,'Wrong GPU')
    assert config['gpu_uuid']=='GPU-cd81f73e-e327-2f98-5828-97cf88152d23'
    assert sha(src/'config.json')==(src/'config.sha256').read_text().strip()==meta['config_sha256']
    assert config['source_model_match_v2']
    assert config['design']==design==meta['design'],'Design copies differ from frozen configuration'
    mode=config['mode'];assert mode in ['full','frontier']
    runner=Path(__file__).resolve().parent/('repeated_gpu2.py' if mode=='full' else 'frontier_gpu2.py')
    if sha(runner)!=config['script_sha256']:print('Note: site directories in the runner were replaced by placeholders; the executed runner had SHA-256',config['script_sha256'])
    prior=load(Path(__file__).resolve().parent/'v2_metadata_reference.json')
    assert prior['model_sha256']==meta['model_sha256']
    assert [s['decoded_frames_sha256'] for s in prior['sources']]==config['source_frame_sha256']
    if partial80:
        assert mode=='full'
        summaries=[s for s in summaries if s['fps']==80]
        selected_design=[s for s in design if s['fps']==80]
        assert len(summaries)==len(selected_design)==36
    else:
        assert (src/'COMPLETE').exists(),'Wait for full completion'
        selected_design=design
        assert len(summaries)==len(design)==(72 if mode=='full' else 36)
    assert [s['label'] for s in summaries]==[s['label'] for s in selected_design]
    assert [s['decoded_frames'] for s in meta['sources']]==[158,600,600]
    assert meta['model_sha256']==config['model_sha256']
    assert [s['decoded_frames_sha256'] for s in meta['sources']]==config['source_frame_sha256']
    for c in meta['calibration']:
        assert len(c['samples_ms'])==120
        equal(np.mean(c['samples_ms']),c['mean_ms'],'Calibration mean')
    raw={s['label']:pd.read_csv(src/f"{s['label']}_delivered.csv",keep_default_na=False) for s in summaries}
    windows={}
    for s in summaries:
        key=(s['source'],s['fps'])
        siblings=[x for x in summaries if (x['source'],x['fps'])==key]
        windows[key]=(max(raw[x['label']].finish_ms.iloc[0] for x in siblings),(s['n']-1)*1000/s['fps'])
    metrics=[];telemetry=[];checks=[];sampled_gpu2_pids=None
    for s,planned in zip(summaries,selected_design):
        label=s['label'];d=raw[label];a=pd.read_csv(src/f'{label}_arrivals.csv')
        drop_path=src/f'{label}_dropped.csv'
        drops=pd.read_csv(drop_path) if drop_path.exists() else pd.DataFrame(columns=['frame','reason'])
        ids=d.frame.to_numpy(dtype=int);drop_ids=drops.frame.to_numpy(dtype=int);n=s['n'];fps=s['fps']
        for key in ['label','source','fps','policy','duration_s','order','replicate','admission_fraction']:
            assert s[key]==planned[key],f'{label}: altered design {key}'
        assert len(a)==n==fps*s['duration_s']
        assert np.array_equal(np.sort(np.r_[ids,drop_ids]),np.arange(n)),f'{label}: frame partition'
        assert np.array_equal(a.frame.to_numpy(),np.arange(n))
        assert len(d)==s['delivered'] and len(drops)==s['dropped']
        assert np.all(np.diff(ids)>0) and np.all(np.diff(a.arrival_ms)>=0)
        assert d.error.eq('').all() and s['errors']==0 and (d.n_detections>=0).all()
        assert (d.start_ms>=d.arrival_ms).all() and (d.finish_ms>=d.start_ms).all()
        assert (d.start_ms.to_numpy()[1:]>=d.finish_ms.to_numpy()[:-1]).all(),f'{label}: overlapping serial service'
        equal(a.scheduled_ms,np.arange(n)*1000/fps,f'{label}: schedule')
        equal(d.arrival_ms,a.set_index('frame').loc[ids,'arrival_ms'],f'{label}: arrival linkage')
        equal(d.finish_ms-d.arrival_ms,d.latency_ms,f'{label}: latency identity')
        equal(d.start_ms-d.arrival_ms,d.wait_ms,f'{label}: waiting identity')
        equal(d.finish_ms-d.start_ms,d.service_ms,f'{label}: service identity')
        if s['policy'] in ['latest','drop_newest']:assert s['queue_max']<=1
        if s['policy']=='fifo':assert len(drops)==0
        if mode=='frontier':
            fraction=s['admission_fraction']
            preadmitted=np.array([i for i in range(n) if int((i+1)*fraction+1e-10)>int(i*fraction+1e-10)])
            budget_drops=drops.loc[drops.reason=='periodic_budget','frame'].to_numpy(dtype=int)
            assert np.array_equal(np.sort(budget_drops),np.setdiff1d(np.arange(n),preadmitted)),f'{label}: frozen pre-admission sequence'
            assert np.isin(ids,preadmitted).all(),f'{label}: delivered nonadmitted frame'
        allowed={'fifo':set(),'periodic':{'periodic_budget'},'latest':{'replaced_pending','periodic_budget'},'drop_newest':{'queue_full','periodic_budget'}}[s['policy']]
        assert set(drops.reason)<=allowed,f'{label}: invalid discard reason'
        if s['policy']=='periodic':
            keep=s['admission_fraction'] if mode=='frontier' else min(1.,.95/(fps*meta['calibration'][s['source']]['mean_ms']/1000))
            equal(keep,s['periodic_keep_fraction'],f'{label}: changed budget')
            expected=np.array([i for i in range(n) if int((i+1)*keep+1e-10)>int(i*keep+1e-10)])
            assert np.array_equal(ids,expected),f'{label}: periodic admission changed'
        calculated=dict(p99_ms=float(np.percentile(d.latency_ms,99)),max_ms=float(d.latency_ms.max()),
            mean_service_ms=float(d.service_ms.mean()),drop_pct=100*len(drops)/n,
            miss_pct=float((d.latency_ms>33).mean()*100),max_arrival_lateness_ms=float((a.arrival_ms-a.scheduled_ms).max()))
        for k,v in calculated.items():equal(s[k],v,f'{label}: logged {k}')
        left,right=windows[(s['source'],fps)];age,maxage=independent_age(d,left,right)
        tm=load(src/f'{label}_telemetry.json')
        assert tm and not any(x['unexpected_gpu2_apps'] for x in tm)
        assert not s['foreign_gpu2_process_detected']
        for x in tm:
            app_lines=x['gpu2_compute_apps']
            pids={int(line.split(',')[0]) for line in app_lines}
            assert len(app_lines)==len(pids)==2 and all(config['gpu_uuid'] in line for line in app_lines),f'{label}: missing/incomplete process-query evidence'
            if sampled_gpu2_pids is None:sampled_gpu2_pids=pids
            assert pids==sampled_gpu2_pids,f'{label}: GPU2 process identities changed'
            parts=[v.strip() for v in x['values'].split(',')]
            assert len(parts)==5 and not x['error']
            telemetry.append(dict(label=label,elapsed_s=x['elapsed_s'],timestamp=parts[0],temperature_c=float(parts[1]),power_w=float(parts[2]),utilization_pct=float(parts[3]),memory_mib=float(parts[4])))
        item=dict(s,**{k:v for k,v in calculated.items() if k not in s})
        item.update(mean_aoi_ms=age,max_aoi_ms=maxage,common_window_start_ms=left,common_window_end_ms=right,
            effective_pre_admission_fraction=s['periodic_keep_fraction'] if s['policy']=='periodic' else s['admission_fraction'],
            completion_gap_p99_ms=float(np.percentile(np.diff(d.finish_ms),99)),
            jitter_p99_ms=float(np.percentile(a.arrival_ms-a.scheduled_ms,99)),
            final_finish_ms=float(d.finish_ms.iloc[-1]),drain_after_last_scheduled_ms=max(0,float(d.finish_ms.iloc[-1]-right)),
            delivered_rate_to_last_completion_fps=float(len(d)/(d.finish_ms.iloc[-1]/1000)),
            bookkeeping_inclusive_run_rate_fps=float(len(d)/s['wall_elapsed_s']),
            delivered_within_offer_window=int((d.finish_ms<=right).sum()),
            completion_rate_within_offer_window_fps=float((d.finish_ms<=right).sum()/(right/1000)))
        metrics.append(item);checks.append(dict(label=label,frame_partition=True,all_timestamp_identities=True,periodic_frozen_budget=True,no_inference_error=True,two_consistent_GPU2_processes_observed_in_every_sample=True,no_foreign_gpu2_process_in_samples=True))
    df=pd.DataFrame(metrics);df.to_csv(out/'all_arm_metrics.csv',index=False)
    pd.DataFrame(telemetry).to_csv(out/'gpu2_telemetry.csv',index=False)
    grouping=['source','fps','policy']+(['admission_fraction'] if mode=='frontier' else [])
    agg=[]
    for key,g in df.groupby(grouping,sort=False):
        assert sorted(g.replicate.tolist())==[1,2,3]
        for metric in ['p99_ms','mean_aoi_ms','drop_pct','miss_pct','mean_service_ms','completion_gap_p99_ms']:
            agg.append(dict(zip(grouping,key),metric=metric,**interval(g[metric])))
    aggregate=pd.DataFrame(agg);aggregate.to_csv(out/'repeated_mean_range_tCI.csv',index=False)
    paired=[];pair_units=['source','fps']+(['admission_fraction'] if mode=='frontier' else [])
    for key,g in df.groupby(pair_units,sort=False):
        pivot=g.set_index(['policy','replicate']);keys=dict(zip(pair_units,key))
        for control in [p for p in POLICIES if p!='latest' and p in g.policy.unique()]:
            lm=pivot.loc['latest'].sort_index();cm=pivot.loc[control].sort_index()
            count_matches=(lm.delivered.to_numpy()==cm.delivered.to_numpy()).tolist()
            for metric in ['p99_ms','mean_aoi_ms','drop_pct']:
                vals=lm[metric].to_numpy()-cm[metric].to_numpy()
                paired.append(dict(keys,comparison='latest_minus_'+control,metric=metric,
                    exact_retained_count_match_replicates=sum(count_matches),all_replicates_exact_count_match=all(count_matches),
                    **interval(vals)))
    pd.DataFrame(paired).to_csv(out/'paired_run_differences.csv',index=False)
    total=dict(mode=mode,provisional_80fps_only=partial80,arms=len(df),scheduled=int(df.n.sum()),
        delivered=int(df.delivered.sum()),dropped=int(df.dropped.sum()),errors=int(df.errors.sum()),
        physical_gpu_count=1,fresh_repetitions=3,source_count=int(df.source.nunique()),
        inference_limit='Conditional within one GPU. Repeats are runs, not frames, farms or independent GPUs. t intervals use df2; mean/range are primary descriptive summaries.',
        loss_limit='Equal offered pre-admission fractions do not ensure equal final loss. Only exact retained-count matches are flagged; no causal equal-loss superiority is inferred.',
        logger_condition='Ultralytics emits one half-deprecation warning per predict call, as in v2; logged timing includes this software path.',
        throughput_definitions='delivered_rate_to_last_completion_fps uses producer origin to final recorded completion; completion_rate_within_offer_window_fps uses origin to last scheduled arrival; bookkeeping_inclusive_run_rate_fps additionally includes monitor join/file writes and is not inference throughput.',
        sampled_gpu2_pids=sorted(sampled_gpu2_pids),
        process_monitor_limit='Two consistent expected GPU2 processes were present in every saved sample and no additional process was observed. Samples are about two seconds apart; process-query stderr/returncode were not stored, so continuous process isolation is not proved.',
        config_sha256=sha(src/'config.json'),model_sha256=meta['model_sha256'],source_frame_sha256=config['source_frame_sha256'],
        analysis_script_sha256=sha(Path(__file__)),analysis_environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
        GPU2_temperature_range_c=[min(x['temperature_c'] for x in telemetry),max(x['temperature_c'] for x in telemetry)],
        checks=checks)
    assert total['scheduled']==total['delivered']+total['dropped']
    (out/'verification_summary.json').write_text(json.dumps(total,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    if mode=='full':
        rates=sorted(df.fps.unique(),reverse=True)
        fig,axes=plt.subplots(len(rates),3,figsize=(11,3.3*len(rates)),squeeze=False)
        ylow=10**math.floor(math.log10(df.p99_ms.min()))
        highstep=10**math.floor(math.log10(df.p99_ms.max()))
        yhigh=math.ceil(df.p99_ms.max()*1.1/highstep)*highstep
        for row,fps in enumerate(rates):
            for source in range(3):
                ax=axes[row,source]
                for pos,policy in enumerate(POLICIES):
                    vals=df.query('source==@source and fps==@fps and policy==@policy').sort_values('replicate').p99_ms.to_numpy()
                    ax.scatter(pos+np.array([-.10,0,.10]),vals,color=COLORS[policy],s=28)
                    ax.plot([pos-.18,pos+.18],[vals.mean(),vals.mean()],color=COLORS[policy],lw=2)
                ax.set_xticks(range(4),['FIFO','Latest','Drop\nnewest','Periodic']);ax.set_yscale('log');ax.grid(axis='y',alpha=.2)
                ax.set_ylim(ylow,yhigh)
                ax.axhline(33,color='#888888',ls='--',lw=.8)
                ax.set_title(f'{SOURCE_NAMES[source]} - {fps} arrivals/s')
                if source==0:ax.set_ylabel('Delivered latency p99 (ms)')
        fig.suptitle('Three fresh executions per condition on one GPU',fontsize=13)
        fig.legend(handles=[Line2D([0],[0],color='#888888',ls='--',lw=.8,label='Fixed 33 ms comparison threshold')],loc='lower center',bbox_to_anchor=(.5,-.02),frameon=False)
        fig.tight_layout()
        save_figure(fig,out/'repeated_latency')
        plt.close(fig)
    else:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,metric,title in zip(axes,['p99_ms','mean_aoi_ms'],['Delivered latency p99 (ms)','Mean information age (ms)']):
            for policy in ['latest','drop_newest','periodic']:
                g=df[df.policy==policy].groupby('admission_fraction').agg(drop=('drop_pct','mean'),value=(metric,'mean'))
                ax.plot(g['drop'],g['value'],'o-',color=COLORS[policy],label=NAMES[policy])
                rawp=df[df.policy==policy];ax.scatter(rawp.drop_pct,rawp[metric],s=15,alpha=.35,color=COLORS[policy])
            ax.set_yscale('log');ax.set_xlabel('Actual discarded arrivals (%)');ax.set_ylabel(title);ax.grid(alpha=.2)
        axes[0].legend(frameon=False);fig.suptitle('June source: fixed admission-fraction sensitivity (three runs each)',fontsize=12)
        fig.tight_layout()
        save_figure(fig,out/'loss_latency_sensitivity')
        plt.close(fig)
    manifest=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]
    (out/'analysis_output_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in total.items() if k!='checks'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--partial80',action='store_true')
    a=p.parse_args();run(a.input,a.output,partial80=a.partial80)
