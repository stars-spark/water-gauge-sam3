import sys, types, os, numpy as np
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
import torch
from pycocotools.coco import COCO
from batch_infer_sam_json import SAM3LoRABatchInference
BASE="/media/jiale/AI_Local/水尺水位测量系统"
test_dir=f"{BASE}/02_数据集/waterline_v1_coco/test"
adl=f"{BASE}/03_云端SAM3/autodl_training_backup/SAM3_LoRA-main"
coco=COCO(f"{test_dir}/_annotations.coco.json")
def gtm(img_id,h,w,cid):
    m=np.zeros((h,w),bool)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id,catIds=[cid])): m|=coco.annToMask(a).astype(bool)
    return m
inf=SAM3LoRABatchInference(config_path=f"{adl}/configs/full_lora_config.yaml",
   weights_path=f"{adl}/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt",
   checkpoint_path="checkpoints/sam3.pt", resolution=1008, threshold=0.4, device="cuda")
print("\n方法：prompt='Gauge_Air' → 水位线y = 干区掩码最下沿；GT水位线 = GT_Gauge_Air最下沿")
errs=[]; errs_pct=[]
for im in coco.loadImgs(coco.getImgIds()):
    p=os.path.join(test_dir,im['file_name'])
    if not os.path.exists(p): continue
    h,w=im['height'],im['width']
    r=inf.process_image(p,["Gauge_Air"])
    ga=gtm(im['id'],h,w,1)
    if r is None or ga.sum()==0: print(f"  {im['file_name'][:22]:22s} 跳过"); continue
    pred=np.zeros((h,w),bool)
    for mk in r['masks']: pred|=mk.astype(bool)
    rows=np.where(pred.any(axis=1))[0]; pred_wl=int(rows.max())          # 干区底沿
    grows=np.where(ga.any(axis=1))[0]; gt_wl=int(grows.max())
    gh=grows.max()-grows.min()+1                                          # GT干区高度≈水尺可见高
    err=abs(pred_wl-gt_wl); pct=err/gh*100; errs.append(err); errs_pct.append(pct)
    print(f"  {im['file_name'][:22]:22s} 水位线y pred={pred_wl:4d} gt={gt_wl:4d} 误差={err:4d}px ({pct:.1f}%水尺高)")
if errs:
    print(f"\n>>> 水位线Y定位误差: 中位={np.median(errs):.0f}px 均值={np.mean(errs):.0f}px")
    print(f">>> 占水尺可见高百分比: 中位={np.median(errs_pct):.1f}% 均值={np.mean(errs_pct):.1f}% (n={len(errs)})")
