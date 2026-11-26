"""Independent Linux/RTX 5090 replication; never overwrite the historical runs.
Run with CUDA_VISIBLE_DEVICES=1. Frames/model are existing local research assets.
FIFO vs historical pre-append admission vs strict latest-pending admission.
All arms retain errors and exact timestamps; no measurement is simulated.
"""
import os, time, json, csv, threading, multiprocessing as mp, platform, hashlib
from pathlib import Path
from collections import deque
import numpy as np

ROOT=Path('<HOME>/hanwoo_eval')
OUT=ROOT/'submission_20261026'/os.environ.get('P2_RUN_NAME','p2_live_v2')
MODEL=ROOT/'p2_pilot_20260809/hanwoo-system/detector_baseline/models/best.pt'
N=180; T=1/float(os.environ.get('P2_FPS','30')); D=.033

def tenant(stop, ready, dim):
    import torch
    torch.set_num_threads(1)
    a=torch.randn(dim,dim,device='cuda',dtype=torch.float32)
    b=torch.randn_like(a); c=torch.empty_like(a)
    for _ in range(3): torch.mm(a,b,out=c)
    torch.cuda.synchronize(); ready.set()
    while not stop.is_set():
        torch.mm(a,b,out=c); torch.cuda.synchronize()

def main():
    import torch, cv2, ultralytics
    from ultralytics import YOLO
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    files=sorted((ROOT/'images').glob('*.jpg'))[:64]
    frames=[cv2.imread(str(p)) for p in files]
    assert len(frames)==64 and all(x is not None for x in frames)
    model=YOLO(str(MODEL))
    def infer(i):
        result=model.predict(frames[i%len(frames)],imgsz=640,device=0,half=False,verbose=False)[0]
        boxes=result.boxes.xyxy.cpu().numpy()
        # Deterministic CPU centroid/boundary-state proxy; no DB or network I/O.
        if len(boxes):
            xy=(boxes[:,:2]+boxes[:,2:])/2
            _=int(np.sum((xy[:,0]<50)|(xy[:,1]<50)))
        torch.cuda.synchronize()
        return len(boxes)
    for i in range(20): infer(i)
    def calibration(k=30):
        vals=[]
        for i in range(k):
            st=time.perf_counter(); infer(i); vals.append((time.perf_counter()-st)*1000)
        return dict(mean_ms=float(np.mean(vals)),p99_ms=float(np.percentile(vals,99)),values_ms=vals)
    idle=calibration()
    trials=[]; stop=ready=proc=None
    for dim in [4096,8192,12288,16384]:
        stop=mp.Event(); ready=mp.Event()
        proc=mp.Process(target=tenant,args=(stop,ready,dim));proc.start()
        if not ready.wait(60):
            stop.set();proc.join(15)
            raise RuntimeError('Co-tenant failed readiness')
        time.sleep(1)
        cal=calibration(); trials.append(dict(dim=dim,**cal))
        print('calibration',dim,cal['mean_ms'],flush=True)
        if cal['mean_ms']>=37 or dim==16384: break
        stop.set();proc.join(15)
        assert not proc.is_alive()
    metadata=dict(date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        platform=platform.platform(),torch=torch.__version__,ultralytics=ultralytics.__version__,
        gpu=torch.cuda.get_device_name(0),model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        model=str(MODEL),frames=[str(p.name) for p in files],n_frames_per_arm=N,
        arrival_fps=1/T,deadline_ms=33,precision='FP32',imgsz=640,idle=idle,
        calibration_trials=trials,chosen_dim=dim,policy='6 permutations of 3 arms; no result exclusions')
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    orders=[['fifo','legacy','latest'],['legacy','latest','fifo'],['latest','fifo','legacy'],
            ['fifo','latest','legacy'],['latest','legacy','fifo'],['legacy','fifo','latest']]
    summaries=[]
    try:
        for rep,order in enumerate(orders,1):
            for arm in order:
                q=deque(); cv=threading.Condition(); done=False; rows=[]; dropped=[]
                state={'last_completion':time.perf_counter(),'estimate':trials[-1]['mean_ms']/1000,'max_queue':0}
                start=time.perf_counter()
                def producer():
                    nonlocal done
                    for i in range(N):
                        delay=start+i*T-time.perf_counter()
                        if delay>0: time.sleep(delay)
                        at=time.perf_counter()
                        with cv:
                            if arm=='legacy':
                                while len(q)>1 and max(state['last_completion']-q[0][0],0)+state['estimate']>D:
                                    dropped.append(q.popleft()[1])
                            q.append((at,i))
                            if arm=='latest':
                                while len(q)>1: dropped.append(q.popleft()[1])
                            state['max_queue']=max(state['max_queue'],len(q));cv.notify()
                    with cv: done=True;cv.notify()
                thread=threading.Thread(target=producer);thread.start()
                while True:
                    with cv:
                        while not q and not done: cv.wait()
                        if not q and done: break
                        at,i=q.popleft()
                    st=time.perf_counter();error='';ndet=-1
                    try: ndet=infer(i)
                    except Exception as exc: error=repr(exc)
                    end=time.perf_counter()
                    with cv:
                        state['last_completion']=end
                        state['estimate']=.8*(end-st)+.2*state['estimate']
                    rows.append(dict(frame=i,arrival_ms=(at-start)*1000,start_ms=(st-start)*1000,
                        finish_ms=(end-start)*1000,wait_ms=(st-at)*1000,service_ms=(end-st)*1000,
                        latency_ms=(end-at)*1000,n_detections=ndet,error=error))
                thread.join()
                assert len(rows)+len(dropped)==N
                assert len({r['frame'] for r in rows}|set(dropped))==N
                if arm=='latest': assert state['max_queue']<=1
                with (OUT/f'rep{rep}_{arm}.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
                latency=np.array([r['latency_ms'] for r in rows]);service=np.array([r['service_ms'] for r in rows])
                summary=dict(rep=rep,arm=arm,order=order.index(arm)+1,delivered=len(rows),dropped=len(dropped),
                    dropped_ids=dropped,errors=sum(bool(r['error']) for r in rows),queue_max=state['max_queue'],
                    p99_ms=float(np.percentile(latency,99)),max_ms=float(max(latency)),
                    mean_service_ms=float(service.mean()),rho=float(service.mean()/1000/T),
                    miss_pct=float(100*np.mean(latency>33)),drop_pct=100*len(dropped)/N)
                summaries.append(summary)
                (OUT/'summaries.json').write_text(json.dumps(summaries,indent=2))
                print('arm',rep,arm,'p99',round(summary['p99_ms'],2),'rho',round(summary['rho'],2),'drop',len(dropped),flush=True)
                time.sleep(.5)
    finally:
        stop.set();proc.join(30)
    (OUT/'COMPLETE').write_text('All 18 arms completed; exact timestamps and failures retained.\n')

if __name__=='__main__':
    mp.set_start_method('spawn');main()
