"""Prospectively fixed repeated P2 benchmark. GPU 2 exclusively.
No raw media or learned weights are redistributed. Native video time is not
the artificial arrival clock; event-window retention is a separate diagnostic.
"""
import os, json, csv, time, threading, multiprocessing as mp, hashlib, platform, subprocess, argparse, sys
from pathlib import Path
from collections import deque
import numpy as np

ROOT=Path('<HOME>/hanwoo_eval')
SOURCE_ROOT=ROOT/'submission_20261026_v2/p2'
OUT=ROOT/'submission_20261026_v4/p2'
RUN=OUT/'repeated_gpu2'
GPU_INDEX=2
EXPECTED_GPU_UUID='GPU-cd81f73e-e327-2f98-5828-97cf88152d23'
CONFIG = {}
MODEL=ROOT/'p2_pilot_20260809/hanwoo-system/detector_baseline/models/best.pt'
SOURCES=[SOURCE_ROOT/'cows2021_first_clip.avi',
 Path('<DATA_ROOT>/NVR_20250620000000_20250620190600/ch01_20250620130000.mp4'),
 Path('<DATA_ROOT>/NVR_20251101080000_20251101165959/ch02_20251101080000.mp4')]
POLICIES=['fifo','latest','drop_newest','periodic']

def tenant(stop,ready,burst):
    import torch
    torch.manual_seed(20261026)
    torch.set_num_threads(1)
    a=torch.randn(16384,16384,device='cuda'); b=torch.randn_like(a); c=torch.empty_like(a)
    for _ in range(3): torch.mm(a,b,out=c)
    torch.cuda.synchronize(); ready.set(); origin=time.perf_counter()
    while not stop.is_set():
        if burst.is_set() and int((time.perf_counter()-origin)/5)%2:
            stop.wait(.01); continue
        torch.mm(a,b,out=c); torch.cuda.synchronize()

def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    global RUN, CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'full', 'frontier'], default='full')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2', 'GPU2-only visibility required'
    if args.mode == 'smoke':
        RUN = OUT/'smoke_gpu2'
    elif args.mode == 'frontier':
        RUN = OUT/'frontier_gpu2'
    # Inspect only compute-app metadata; never terminate unrelated work.
    snapshot = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    assert EXPECTED_GPU_UUID not in snapshot, 'GPU2 is already in use; do not share with unrelated processes'
    import torch, cv2, ultralytics
    from ultralytics import YOLO
    RUN.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    torch.manual_seed(20261026); np.random.seed(20261026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.device_count() == 1
    (RUN/'preflight_compute_snapshot.txt').write_text(snapshot)
    arrays=[];source_info=[]
    for source in SOURCES:
        cap=cv2.VideoCapture(str(source));frames=[];pts=[]
        fps=cap.get(cv2.CAP_PROP_FPS);total=cap.get(cv2.CAP_PROP_FRAME_COUNT)
        while len(frames)<600:
            ok,im=cap.read()
            if not ok:break
            frames.append(cv2.resize(im,(640,360)));pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        cap.release();assert frames
        digest=hashlib.sha256()
        for frame in frames:digest.update(frame.tobytes())
        arrays.append(frames)
        source_info.append(dict(path=str(source),file_bytes=source.stat().st_size,
            container_reported_fps=fps,container_frame_count=total,decoded_frames=len(frames),
            decoded_frames_sha256=digest.hexdigest(),decoder_pts_ms=pts))
    prior=json.loads((SOURCE_ROOT/'extended_live/metadata.json').read_text())
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==prior['model_sha256'], 'checkpoint drift'
    for now,old in zip(source_info,prior['sources']):
        assert now['decoded_frames_sha256']==old['decoded_frames_sha256'], 'decoded frame drift'
    duration = 5 if args.mode=='smoke' else 60
    rng=np.random.default_rng(20261026)
    design=[]
    if args.mode=='smoke':
        for j,policy in enumerate(POLICIES):
            design.append(dict(label=f'smoke_s1_80_{policy}',source=1,fps=80,policy=policy,duration_s=duration,order=j+1,replicate=0,admission_fraction=1.0))
    elif args.mode=='full':
        # Complete all three 80-fps repeats first; then three 30-fps controls.
        for fps in [80,30]:
            for rep in range(1,4):
                cells=[(s,p) for s in range(3) for p in POLICIES]
                for idx in rng.permutation(len(cells)):
                    source_id,policy=cells[int(idx)]
                    design.append(dict(label=f'r{rep}_s{source_id}_{fps}_{policy}',source=source_id,fps=fps,policy=policy,duration_s=duration,order=len(design)+1,replicate=rep,admission_fraction=1.0))
    else:
        # A separate, pre-specified sensitivity grid, not tuned to evaluation loss.
        # On June source only: common deterministic pre-admission fractions,
        # followed by latest / drop-newest / FIFO queueing. Periodic is FIFO
        # after deterministic admission and is labelled periodic here.
        for rep in range(1,4):
            cells=[(p,f) for p in ['latest','drop_newest','periodic'] for f in [.5,.65,.8,.95]]
            for idx in rng.permutation(len(cells)):
                policy,fraction=cells[int(idx)]
                design.append(dict(label=f'frontier_r{rep}_s1_80_{policy}_f{int(fraction*100)}',source=1,fps=80,policy=policy,duration_s=duration,order=len(design)+1,replicate=rep,admission_fraction=fraction))
    CONFIG=dict(mode=args.mode,seed=20261026,physical_gpu_index=GPU_INDEX,gpu_uuid=EXPECTED_GPU_UUID,
        source_model_match_v2=True,design=design,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_sha256=prior['model_sha256'],source_frame_sha256=[x['decoded_frames_sha256'] for x in source_info],
        precision='FP32',tf32_matmul=False,tf32_cudnn=True,tenant_matrix_dimension=16384,
        calibration_n_per_source=120,pre_tenant_warmup=30,per_arm_warmup=0,
        limitation='Same host/GPU across all repeats; RAM replay; host CPU/I/O can be shared; source loops are not independent farms.')
    config_text=json.dumps(CONFIG,sort_keys=True,indent=2)
    (RUN/'config.json').write_text(config_text)
    (RUN/'config.sha256').write_text(hashlib.sha256(config_text.encode()).hexdigest()+'\n')
    (RUN/'design.json').write_text(json.dumps(design,indent=2))
    model=YOLO(str(MODEL))
    def infer(frame):
        result=model.predict(frame,imgsz=640,device=0,half=False,batch=1,conf=.25,iou=.7,max_det=300,verbose=False)[0]
        boxes=result.boxes.xyxy.cpu().numpy()
        if len(boxes):
            xy=(boxes[:,:2]+boxes[:,2:])/2
            _=int(np.sum((xy[:,0]<50)|(xy[:,1]<50)))
        torch.cuda.synchronize();return len(boxes)
    for i in range(30):infer(arrays[i%3][i%len(arrays[i%3])])
    stop,ready,burst=mp.Event(),mp.Event(),mp.Event()
    proc=mp.Process(target=tenant,args=(stop,ready,burst));proc.start()
    if not ready.wait(60):raise RuntimeError('Co-tenant startup failed')
    summaries=[]
    try:
        calibration=[]
        for source_id,frames in enumerate(arrays):
            times=[]
            for i in range(120):
                st=time.perf_counter();infer(frames[i%len(frames)]);times.append((time.perf_counter()-st)*1000)
            calibration.append(dict(source=source_id,mean_ms=float(np.mean(times)),p90_ms=float(np.percentile(times,90)),samples_ms=times))
        meta=dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),platform=platform.platform(),
            torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(0),
            model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),sources=source_info,
            input_shape=[360,640],imgsz=640,precision='FP32',conf=.25,iou=.7,max_det=300,
            calibration=calibration,periodic_rule=('prespecified deterministic fractions [0.50, 0.65, 0.80, 0.95]; separate calibration is recorded but does not tune the grid' if args.mode=='frontier' else 'min(1, 0.95/(fps*mean_service_seconds)); independent 120-frame calibration per source'),
            design=CONFIG['design'], physical_gpu_index=GPU_INDEX, gpu_uuid=EXPECTED_GPU_UUID,
            config_sha256=hashlib.sha256((RUN/'config.json').read_bytes()).hexdigest(),
            torch_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,torch_tf32_cudnn=torch.backends.cudnn.allow_tf32,
            warning='Artificial replay arrival times. Repeated decoded frames. No annotated animal-event accuracy.')
        (RUN/'metadata.json').write_text(json.dumps(meta,indent=2))
        def run_arm(label,source_id,fps,policy,duration,order,replicate,admission_fraction):
            frames=arrays[source_id];n=int(fps*duration);interval=1/fps
            keep=admission_fraction if args.mode=='frontier' else min(1.,.95/(fps*calibration[source_id]['mean_ms']/1000))
            arm_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            (RUN/'current_arm.json').write_text(json.dumps(dict(label=label,started_utc=arm_started_utc,order=order)))
            q=deque();cv=threading.Condition();done=False;rows=[];arrivals=[];discard=[]
            state={'max_queue':0};origin=time.perf_counter();telemetry=[];monitor_stop=threading.Event()
            def monitor():
                while not monitor_stop.is_set():
                    value=subprocess.run(['nvidia-smi','--id=2','--query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
                    gpu2apps=[line for line in apps.stdout.splitlines() if EXPECTED_GPU_UUID in line]
                    unexpected=[line for line in gpu2apps if int(line.split(',')[0]) not in {os.getpid(),proc.pid}]
                    telemetry.append(dict(elapsed_s=time.perf_counter()-origin,values=value.stdout.strip(),error=value.stderr.strip(),gpu2_compute_apps=gpu2apps,unexpected_gpu2_apps=unexpected))
                    monitor_stop.wait(2)
            mon=threading.Thread(target=monitor);mon.start()
            def producer():
                nonlocal done
                try:
                    for i in range(n):
                        delay=origin+i*interval-time.perf_counter()
                        if delay>0:time.sleep(delay)
                        at=time.perf_counter()
                        arrivals.append(dict(frame=i,arrival_ms=(at-origin)*1000,scheduled_ms=i*interval*1000))
                        with cv:
                            active_keep=keep if policy=='periodic' else admission_fraction
                            if (policy=='periodic' or admission_fraction<1) and int((i+1)*active_keep+1e-10)==int(i*active_keep+1e-10):
                                discard.append(dict(frame=i,reason='periodic_budget'));continue
                            if policy=='drop_newest' and len(q)>=1:
                                discard.append(dict(frame=i,reason='queue_full'));continue
                            q.append((at,i))
                            if policy=='latest':
                                while len(q)>1:discard.append(dict(frame=q.popleft()[1],reason='replaced_pending'))
                            state['max_queue']=max(state['max_queue'],len(q));cv.notify()
                finally:
                    with cv:done=True;cv.notify_all()
            thread=threading.Thread(target=producer);thread.start()
            while True:
                with cv:
                    while not q and not done:cv.wait()
                    if not q and done:break
                    at,i=q.popleft()
                st=time.perf_counter();err='';ndet=-1
                try:ndet=infer(frames[i%len(frames)])
                except Exception as exc:err=repr(exc)
                end=time.perf_counter()
                rows.append(dict(frame=i,arrival_ms=(at-origin)*1000,start_ms=(st-origin)*1000,
                    finish_ms=(end-origin)*1000,wait_ms=(st-at)*1000,service_ms=(end-st)*1000,
                    latency_ms=(end-at)*1000,n_detections=ndet,error=err))
            thread.join();monitor_stop.set();mon.join()
            assert len(rows)+len(discard)==len(arrivals)==n
            assert len({r['frame'] for r in rows}|{r['frame'] for r in discard})==n
            assert len({r['frame'] for r in rows})==len(rows)
            if policy in ['latest','drop_newest']:assert state['max_queue']<=1
            write_csv(RUN/f'{label}_delivered.csv',rows);write_csv(RUN/f'{label}_arrivals.csv',arrivals)
            if discard:write_csv(RUN/f'{label}_dropped.csv',discard)
            (RUN/f'{label}_telemetry.json').write_text(json.dumps(telemetry,indent=2))
            latency=np.array([r['latency_ms'] for r in rows])
            item=dict(label=label,source=source_id,fps=fps,policy=policy,duration_s=duration,order=order,
                replicate=replicate,admission_fraction=admission_fraction,started_utc=arm_started_utc,
                wall_elapsed_s=time.perf_counter()-origin,foreign_gpu2_process_detected=any(t['unexpected_gpu2_apps'] for t in telemetry),
                n=n,delivered=len(rows),dropped=len(discard),drop_pct=100*len(discard)/n,
                errors=sum(bool(r['error']) for r in rows),queue_max=state['max_queue'],
                p99_ms=float(np.percentile(latency,99)),max_ms=float(np.max(latency)),
                mean_service_ms=float(np.mean([r['service_ms'] for r in rows])),
                miss_pct=float(np.mean(latency>33)*100),periodic_keep_fraction=keep,
                max_arrival_lateness_ms=max(r['arrival_ms']-r['scheduled_ms'] for r in arrivals))
            summaries.append(item)
            (RUN/'summaries.json').write_text(json.dumps(summaries,indent=2))
            print(json.dumps(item),flush=True)
            if item['errors']:raise RuntimeError('Inference errors retained; stop for inspection')
            if item['foreign_gpu2_process_detected']:raise RuntimeError('Unrelated GPU2 process detected; retained run for inspection')
        for cell in design:
            run_arm(cell['label'],cell['source'],cell['fps'],cell['policy'],cell['duration_s'],cell['order'],cell['replicate'],cell['admission_fraction'])
        (RUN/'COMPLETE').write_text(f"Completed {len(design)} fresh arms in mode {args.mode}.\n")
    finally:
        stop.set();proc.join(30)
        if proc.is_alive():proc.terminate();proc.join()

if __name__=='__main__':
    mp.set_start_method('spawn');main()
