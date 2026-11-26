"""Prospectively fixed extended P2 benchmark. GPU 1 exclusively.
No raw media or learned weights are redistributed. Native video time is not
the artificial arrival clock; event-window retention is a separate diagnostic.
"""
import os, json, csv, time, threading, multiprocessing as mp, hashlib, platform, subprocess
from pathlib import Path
from collections import deque
import numpy as np

ROOT=Path('<HOME>/hanwoo_eval')
OUT=ROOT/'submission_20261026_v2/p2'
RUN=OUT/'extended_live'
MODEL=ROOT/'p2_pilot_20260809/hanwoo-system/detector_baseline/models/best.pt'
SOURCES=[OUT/'cows2021_first_clip.avi',
 Path('<DATA_ROOT>/NVR_20250620000000_20250620190600/ch01_20250620130000.mp4'),
 Path('<DATA_ROOT>/NVR_20251101080000_20251101165959/ch02_20251101080000.mp4')]
POLICIES=['fifo','latest','drop_newest','periodic']

def tenant(stop,ready,burst):
    import torch
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
    import torch, cv2, ultralytics
    from ultralytics import YOLO
    RUN.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);cv2.setNumThreads(1)
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
            calibration=calibration,periodic_rule='min(1, 0.95/(fps*mean_service_seconds)); independent 120-frame calibration per source',
            design='3 source blocks x 2 arrival rates x 4 policies x 60s; fixed seed order. Plus latest 600s burst workload.',
            warning='Artificial replay arrival times. Repeated decoded frames. No annotated animal-event accuracy.')
        (RUN/'metadata.json').write_text(json.dumps(meta,indent=2))
        def run_arm(label,source_id,fps,policy,duration,order):
            frames=arrays[source_id];n=int(fps*duration);interval=1/fps
            keep=min(1.,.95/(fps*calibration[source_id]['mean_ms']/1000))
            q=deque();cv=threading.Condition();done=False;rows=[];arrivals=[];discard=[]
            state={'max_queue':0};origin=time.perf_counter();telemetry=[];monitor_stop=threading.Event()
            def monitor():
                while not monitor_stop.is_set():
                    value=subprocess.run(['nvidia-smi','--id=1','--query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
                    telemetry.append(dict(elapsed_s=time.perf_counter()-origin,values=value.stdout.strip(),error=value.stderr.strip()))
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
                            if policy=='periodic' and int((i+1)*keep+1e-10)==int(i*keep+1e-10):
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
        rng=np.random.default_rng(20261026)
        design=[]
        for source_id in range(3):
            for fps in [30,80]:
                order=list(rng.permutation(POLICIES))
                for j,policy in enumerate(order):design.append((f's{source_id}_{fps}_{policy}',source_id,fps,policy,60,j+1))
        design.append(('s1_80_latest_burst600',1,80,'latest',600,1))
        (RUN/'design.json').write_text(json.dumps(design,indent=2))
        for args in design:
            if args[4]==600:burst.set()
            run_arm(*args)
        (RUN/'COMPLETE').write_text('All 24 one-minute arms and one ten-minute burst arm completed.\n')
    finally:
        stop.set();proc.join(30)
        if proc.is_alive():proc.terminate();proc.join()

if __name__=='__main__':
    mp.set_start_method('spawn');main()
