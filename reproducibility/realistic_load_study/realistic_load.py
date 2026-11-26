"""Multi-camera native-rate load, heavier detector and training co-tenant benchmark. See protocol.md.

No raw media or learned weights are redistributed. Arrival times are artificial replay times.

Usage (measured stages need the GPU to be free of unrelated compute processes):
  CUDA_VISIBLE_DEVICES=<n> python realistic_load.py --mode smoke|stage1|stage2 --out NEW_DIR \
      --nvr-dir DIR --model-a best.pt --model-b x1280_best.pt --train-yaml data.yaml --train-cwd DIR \
      --prior-metadata metadata.json --protocol protocol.md --gpu-uuid GPU-...
  python realistic_load.py --mode dry --out NEW_DIR        # queue-logic test with simulated service, no GPU
"""
import os, json, csv, time, threading, multiprocessing as mp, hashlib, platform, subprocess, argparse, sys, signal
from pathlib import Path
from collections import deque
import numpy as np

CHANNELS = ['ch02', 'ch01', 'ch03', 'ch04', 'ch05']
POLICIES = ['fifo', 'latest', 'drop_newest', 'periodic']
SEED = 20261115
CELLS_A = [(1, 15), (3, 15), (5, 15), (1, 30), (3, 30), (5, 30)]
CELLS_B = [(1, 15), (1, 30)]
CELLS_T = [(5, 15), (3, 30)]

TRAIN_SCRIPT = r'''
import sys
from ultralytics import YOLO
m = YOLO("yolov8n.yaml")
m.train(data=sys.argv[1], imgsz=640, epochs=100000, batch=16, workers=8, seed=0, device=0, amp=True, val=False, plots=False,
        patience=100000, save=False, project=sys.argv[2], name="tenant", exist_ok=True, verbose=False)
'''


def synthetic_tenant(stop, ready):
    import torch
    torch.manual_seed(20261026); torch.set_num_threads(1)
    a = torch.randn(16384, 16384, device='cuda'); b = torch.randn_like(a); c = torch.empty_like(a)
    for _ in range(3): torch.mm(a, b, out=c)
    torch.cuda.synchronize(); ready.set()
    while not stop.is_set():
        torch.mm(a, b, out=c); torch.cuda.synchronize()


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def build_design(mode):
    rng = np.random.default_rng(SEED); design = []
    def add(prefix, rep, det, k, r, policy, dur):
        design.append(dict(label=f'{prefix}_r{rep}_{det}_k{k}_{r}_{policy}', detector=det, cameras=k, rate=r, policy=policy, duration_s=dur, order=len(design) + 1, replicate=rep))
    if mode in ('smoke', 'dry'):
        for det, k, r in [('A', 5, 15), ('A', 3, 30), ('B', 1, 30)]:
            for p in POLICIES: add(mode, 0, det, k, r, p, 5)
    elif mode == 'stage1':
        for rep in (1, 2, 3):
            cells = [('A', k, r, p) for k, r in CELLS_A for p in POLICIES] + [('B', k, r, p) for k, r in CELLS_B for p in POLICIES]
            for idx in rng.permutation(len(cells)): add('s1', rep, *cells[int(idx)], 60)
    else:
        for rep in (1, 2, 3):
            cells = [('A', k, r, p) for k, r in CELLS_T for p in POLICIES]
            for idx in rng.permutation(len(cells)): add('s2', rep, *cells[int(idx)], 60)
    return design


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['dry', 'smoke', 'stage1', 'stage2'], required=True); ap.add_argument('--out', required=True)
    for k in ('nvr-dir', 'model-a', 'model-b', 'train-yaml', 'train-cwd', 'prior-metadata', 'protocol', 'gpu-uuid'): ap.add_argument('--' + k)
    ap.add_argument('--smoke-tenant', choices=['synthetic', 'training'], default='synthetic')
    ap.add_argument('--allow-shared', action='store_true', help='functional test only; results are not valid measurements')
    ap.add_argument('--start-order', type=int, default=1, help='resume a stage: run only design cells whose order is at least this value')
    a = ap.parse_args(); RUN = Path(a.out); dry = a.mode == 'dry'
    tenant_kind = 'training' if a.mode == 'stage2' or (a.mode == 'smoke' and a.smoke_tenant == 'training') else 'synthetic'
    design = build_design(a.mode)
    if tenant_kind == 'training' and a.mode == 'smoke': design = [c for c in design if c['detector'] == 'A']
    full_design_n = len(design); design = [c for c in design if c['order'] >= a.start_order]      # the order of the full design is unchanged
    snapshot = ''
    if not dry:
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES'); assert gpu and ',' not in gpu, 'exactly one visible GPU required'
        snapshot = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
        if not a.allow_shared: assert a.gpu_uuid not in snapshot, 'GPU already in use by another process'
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / 'preflight_compute_snapshot.txt').write_text(snapshot)
    rng_frames = np.random.default_rng(0)
    if dry:
        arrays = {('A', c): [np.zeros((4, 4, 3), np.uint8)] * 600 for c in range(5)}; arrays[('B', 0)] = arrays[('A', 0)]; source_info = []
        def infer(det, frame):
            time.sleep(0.013 if det == 'A' else 0.045); return int(rng_frames.integers(0, 20))
        model_hash = {}
    else:
        import torch, cv2, ultralytics
        from ultralytics import YOLO
        torch.set_num_threads(1); cv2.setNumThreads(1); torch.manual_seed(SEED); np.random.seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
        assert torch.cuda.device_count() == 1
        arrays, source_info = {}, []
        for c, ch in enumerate(CHANNELS):
            files = sorted(Path(a.nvr_dir).glob(ch + '_*.mp4')); assert len(files) == 1, (ch, files)
            cap = cv2.VideoCapture(str(files[0])); small, big = [], []
            fps = cap.get(cv2.CAP_PROP_FPS); total = cap.get(cv2.CAP_PROP_FRAME_COUNT); w0, h0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            while len(small) < 600:
                ok, im = cap.read()
                if not ok: break
                small.append(cv2.resize(im, (640, 360)))
                if c == 0: big.append(cv2.resize(im, (1280, 720)))
            cap.release(); assert len(small) == 600, (ch, len(small))
            dg = hashlib.sha256()
            for fr in small: dg.update(fr.tobytes())
            arrays[('A', c)] = small
            if c == 0: arrays[('B', 0)] = big
            source_info.append(dict(channel=ch, file=files[0].name, file_bytes=files[0].stat().st_size, container_reported_fps=fps, container_frame_count=total,
                                    native_width=w0, native_height=h0, decoded_frames=len(small), decoded_frames_sha256=dg.hexdigest()))
        prior = json.loads(Path(a.prior_metadata).read_text())
        model_hash = {'A': hashlib.sha256(Path(a.model_a).read_bytes()).hexdigest(), 'B': hashlib.sha256(Path(a.model_b).read_bytes()).hexdigest()}
        assert model_hash['A'] == prior['model_sha256'], 'detector A checkpoint differs from the earlier benchmark'
        assert source_info[0]['decoded_frames_sha256'] == prior['sources'][2]['decoded_frames_sha256'], 'ch02 frames differ from the earlier benchmark'
        models = {'A': YOLO(a.model_a), 'B': YOLO(a.model_b)}; size = {'A': 640, 'B': 1280}
        def infer(det, frame):
            result = models[det].predict(frame, imgsz=size[det], device=0, half=False, batch=1, conf=.25, iou=.7, max_det=300, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            if len(boxes):
                xy = (boxes[:, :2] + boxes[:, 2:]) / 2; _ = int(np.sum((xy[:, 0] < 50) | (xy[:, 1] < 50)))
            torch.cuda.synchronize(); return len(boxes)
        for i in range(30): infer('A', arrays[('A', i % 5)][i])
        for i in range(30): infer('B', arrays[('B', 0)][i])
    config = dict(mode=a.mode, tenant=tenant_kind, seed=SEED, start_order=a.start_order, full_design_cells=full_design_n, gpu_uuid=a.gpu_uuid, shared_gpu_allowed=bool(a.allow_shared), design=design, channels=CHANNELS,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), protocol_sha256=hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest() if a.protocol else None,
                  model_sha256=model_hash, source_frame_sha256=[s['decoded_frames_sha256'] for s in source_info], precision='FP32', tf32_matmul=False, tf32_cudnn=True,
                  calibration_n=120, pre_tenant_warmup=30, per_arm_warmup=0)
    text = json.dumps(config, sort_keys=True, indent=2); (RUN / 'config.json').write_text(text); (RUN / 'config.sha256').write_text(hashlib.sha256(text.encode()).hexdigest() + '\n')
    # ---------------------------------------------------------------- co-tenant
    stop = proc = tproc = None; allowed = {os.getpid()}
    if not dry and tenant_kind == 'synthetic':
        stop, ready = mp.Event(), mp.Event(); proc = mp.Process(target=synthetic_tenant, args=(stop, ready)); proc.start()
        if not ready.wait(120): raise RuntimeError('co-tenant startup failed')
        allowed.add(proc.pid)
    elif not dry:
        tdir = RUN / 'tenant_training'; tdir.mkdir(); (RUN / 'tenant_train.py').write_text(TRAIN_SCRIPT)
        tproc = subprocess.Popen([sys.executable, str((RUN / 'tenant_train.py').resolve()), a.train_yaml, str(tdir.resolve())], cwd=a.train_cwd,
                                 stdout=(RUN / 'tenant_stdout.log').open('w'), stderr=subprocess.STDOUT, start_new_session=True)
        allowed.add(tproc.pid); results = tdir / 'tenant' / 'results.csv'; t0 = time.time()
        while True:                                                     # wait for one completed training epoch
            if tproc.poll() is not None: raise RuntimeError('training co-tenant exited')
            if results.exists() and len(results.read_text().strip().splitlines()) >= 2: break
            if time.time() - t0 > 900: raise RuntimeError('training co-tenant produced no epoch in 15 minutes')
            time.sleep(2)
    summaries = []
    try:
        calibration = {}
        for key, frames in arrays.items():
            det, cam = key; times = []
            for i in range(120):
                st = time.perf_counter(); infer(det, frames[i % len(frames)]); times.append((time.perf_counter() - st) * 1000)
            calibration[f'{det}{cam}'] = dict(detector=det, camera=cam, mean_ms=float(np.mean(times)), p90_ms=float(np.percentile(times, 90)), samples_ms=times)
        meta = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), platform=platform.platform(), sources=source_info, calibration=calibration,
                    config_sha256=hashlib.sha256((RUN / 'config.json').read_bytes()).hexdigest(), tenant=tenant_kind,
                    periodic_rule='min(1, 0.95/(K*r*mean_service_seconds)); mean over the K cameras of a 120-call calibration per camera',
                    warning='Artificial replay arrival times. Repeated decoded frames. No annotated animal-event accuracy.')
        if not dry: meta.update(torch=torch.__version__, ultralytics=ultralytics.__version__, gpu=torch.cuda.get_device_name(0))
        (RUN / 'metadata.json').write_text(json.dumps(meta, indent=2))

        def run_arm(label, detector, cameras, rate, policy, duration_s, order, replicate):
            K, n_cam = cameras, int(rate * duration_s); interval = 1 / rate; n = n_cam * K
            mean_s = float(np.mean([calibration[f'{detector}{c}']['mean_ms'] for c in range(K)])) / 1000
            keep = min(1., .95 / (K * rate * mean_s)) if policy == 'periodic' else 1.
            started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); (RUN / 'current_arm.json').write_text(json.dumps(dict(label=label, started_utc=started, order=order)))
            schedule = sorted(((i + c / K) * interval, c, i) for c in range(K) for i in range(n_cam))
            fifo = deque(); slot = {}; cv = threading.Condition(); done = False; rows, arrivals, discard = [], [], []; state = {'max_queue': 0}
            telemetry = []; monitor_stop = threading.Event(); origin = time.perf_counter()
            def monitor():
                while not monitor_stop.is_set():
                    if not dry:
                        val = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,uuid,temperature.gpu,power.draw,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory', '--format=csv,noheader'], capture_output=True, text=True)
                        mine = [l for l in apps.stdout.splitlines() if a.gpu_uuid in l]
                        unexpected = [l for l in mine if int(l.split(',')[0]) not in allowed]
                        telemetry.append(dict(elapsed_s=time.perf_counter() - origin, gpu=[l for l in val.stdout.splitlines() if a.gpu_uuid in l], compute_apps=mine, unexpected_apps=unexpected))
                    monitor_stop.wait(2)
            mon = threading.Thread(target=monitor); mon.start()
            def producer():
                nonlocal done
                try:
                    for t, c, i in schedule:
                        delay = origin + t - time.perf_counter()
                        if delay > 0: time.sleep(delay)
                        at = time.perf_counter(); arrivals.append(dict(camera=c, frame=i, arrival_ms=(at - origin) * 1000, scheduled_ms=t * 1000))
                        with cv:
                            if policy == 'periodic' and int((i + 1) * keep + 1e-10) == int(i * keep + 1e-10):
                                discard.append(dict(camera=c, frame=i, reason='periodic_budget')); continue
                            if policy in ('fifo', 'periodic'):
                                fifo.append((at, c, i)); state['max_queue'] = max(state['max_queue'], len(fifo))
                            elif policy == 'drop_newest':
                                if c in slot: discard.append(dict(camera=c, frame=i, reason='queue_full')); continue
                                slot[c] = (at, c, i); state['max_queue'] = max(state['max_queue'], len(slot))
                            else:
                                if c in slot: discard.append(dict(camera=c, frame=slot[c][2], reason='replaced_pending'))
                                slot[c] = (at, c, i); state['max_queue'] = max(state['max_queue'], len(slot))
                            cv.notify()
                finally:
                    with cv: done = True; cv.notify_all()
            th = threading.Thread(target=producer); th.start()
            while True:
                with cv:
                    while not fifo and not slot and not done: cv.wait()
                    if not fifo and not slot and done: break
                    if fifo: at, c, i = fifo.popleft()
                    else:
                        c = min(slot, key=lambda k: slot[k][0]); at, c, i = slot.pop(c)
                st = time.perf_counter(); err = ''; ndet = -1
                try: ndet = infer(detector, arrays[(detector, c)][i % 600])
                except Exception as exc: err = repr(exc)
                end = time.perf_counter()
                rows.append(dict(camera=c, frame=i, arrival_ms=(at - origin) * 1000, start_ms=(st - origin) * 1000, finish_ms=(end - origin) * 1000,
                                 wait_ms=(st - at) * 1000, service_ms=(end - st) * 1000, latency_ms=(end - at) * 1000, n_detections=ndet, error=err))
            th.join(); monitor_stop.set(); mon.join()
            assert len(rows) + len(discard) == len(arrivals) == n
            ids = {(r['camera'], r['frame']) for r in rows}; assert len(ids) == len(rows) and len(ids | {(r['camera'], r['frame']) for r in discard}) == n
            if policy in ('latest', 'drop_newest'): assert state['max_queue'] <= K
            write_csv(RUN / f'{label}_delivered.csv', rows); write_csv(RUN / f'{label}_arrivals.csv', arrivals)
            if discard: write_csv(RUN / f'{label}_dropped.csv', discard)
            (RUN / f'{label}_telemetry.json').write_text(json.dumps(telemetry))
            lat = np.array([r['latency_ms'] for r in rows])
            item = dict(label=label, detector=detector, cameras=K, rate=rate, aggregate_rate=K * rate, policy=policy, duration_s=duration_s, order=order, replicate=replicate,
                        tenant=tenant_kind, started_utc=started, wall_elapsed_s=time.perf_counter() - origin, foreign_process_detected=any(t['unexpected_apps'] for t in telemetry),
                        n=n, delivered=len(rows), dropped=len(discard), drop_pct=100 * len(discard) / n, errors=sum(bool(r['error']) for r in rows), queue_max=state['max_queue'],
                        p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)), p99_ms=float(np.percentile(lat, 99)), max_ms=float(lat.max()),
                        mean_service_ms=float(np.mean([r['service_ms'] for r in rows])), miss33_pct=float(np.mean(lat > 33) * 100), miss_interval_pct=float(np.mean(lat > 1000 * interval) * 100),
                        periodic_keep_fraction=keep, calibration_mean_service_ms=mean_s * 1000, max_arrival_lateness_ms=max(r['arrival_ms'] - r['scheduled_ms'] for r in arrivals))
            summaries.append(item); (RUN / 'summaries.json').write_text(json.dumps(summaries, indent=2)); print(json.dumps(item), flush=True)
            if item['errors']: raise RuntimeError('inference errors; run retained for inspection')
            if item['foreign_process_detected'] and not a.allow_shared: raise RuntimeError('unrelated GPU process detected; run retained for inspection')
            if tproc is not None and tproc.poll() is not None: raise RuntimeError('training co-tenant exited during the stage')

        for cell in design: run_arm(**cell)
        if tproc is not None:
            res = RUN / 'tenant_training' / 'tenant' / 'results.csv'; (RUN / 'tenant_epochs_completed.txt').write_text(str(len(res.read_text().strip().splitlines()) - 1) + '\n')
        (RUN / 'COMPLETE').write_text(f'Completed {len(design)} arms in mode {a.mode}.\n')
    finally:
        if proc is not None:
            stop.set(); proc.join(30)
            if proc.is_alive(): proc.terminate(); proc.join()
        if tproc is not None and tproc.poll() is None:
            os.killpg(os.getpgid(tproc.pid), signal.SIGTERM)
            try: tproc.wait(30)
            except subprocess.TimeoutExpired: os.killpg(os.getpgid(tproc.pid), signal.SIGKILL)


if __name__ == '__main__':
    mp.set_start_method('spawn'); main()
