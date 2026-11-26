"""Read-only postflight source/model/code verification after all 108 arms."""
from pathlib import Path
import hashlib,json,time
import cv2

root=Path('<HOME>/hanwoo_eval/submission_20261026_v4/p2')
assert all((root/d/'COMPLETE').exists() for d in ['repeated_gpu2','frontier_gpu2'])
cv2.setNumThreads(1)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
metas=[json.loads((root/d/'metadata.json').read_text()) for d in ['repeated_gpu2','frontier_gpu2']]
model=Path('<HOME>/hanwoo_eval/p2_pilot_20260809/hanwoo-system/detector_baseline/models/best.pt')
assert sha(model)==metas[0]['model_sha256']==metas[1]['model_sha256']
sources=[]
for s in metas[0]['sources']:
    source=Path(s['path']);capture=cv2.VideoCapture(str(source));digest=hashlib.sha256();n=0
    while n<600:
        ok,frame=capture.read()
        if not ok:break
        digest.update(cv2.resize(frame,(640,360)).tobytes());n+=1
    capture.release()
    assert n==s['decoded_frames'] and digest.hexdigest()==s['decoded_frames_sha256']
    assert source.stat().st_size==s['file_bytes']
    sources.append({'source':len(sources),'file_bytes':source.stat().st_size,'decoded_frames':n,'decoded_frames_sha256':digest.hexdigest(),'matches_before_and_after':True})
runs=[]
for folder,script,logname,arms in [('repeated_gpu2','repeated_gpu2.py','repeated_full.log',72),('frontier_gpu2','frontier_gpu2.py','frontier.log',36)]:
    p=root/folder;config=json.loads((p/'config.json').read_text());summaries=json.loads((p/'summaries.json').read_text())
    assert len(summaries)==arms
    assert sha(root/script)==config['script_sha256']
    assert sha(p/'config.json')==(p/'config.sha256').read_text().strip()
    assert config['source_frame_sha256']==[x['decoded_frames_sha256'] for x in sources]
    log=root/logname;warnings=log.read_bytes().count(b"'half' is deprecated")
    predictions=sum(x['delivered'] for x in summaries)+30+3*120
    assert warnings==predictions,(folder,warnings,predictions)
    runs.append({'folder':folder,'arms':arms,'delivered':sum(x['delivered'] for x in summaries),'config_sha256':sha(p/'config.json'),'runner_sha256':sha(root/script),'log_bytes':log.stat().st_size,'deprecation_warnings':warnings,'predict_calls_including_warmup_and_calibration':predictions,'one_warning_per_predict_verified':True})
result={'verified_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'model_sha256':sha(model),'sources':sources,'runs':runs,'source_contract':'First decoded resized RAM frames, not a checksum of unused whole-video bytes','timing_condition':'Unmodified per-predict deprecation logging as in v2; its overhead is included and not isolated','hardware_scope':'Same physical GPU2 across 108 runs; no cross-device or farm inference'}
out=root/'postflight_verification.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
