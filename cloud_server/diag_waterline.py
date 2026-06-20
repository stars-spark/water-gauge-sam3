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
def iou(a,b):
    u=np.logical_or(a,b).sum(); return float(np.logical_and(a,b).sum())/u if u>0 else 0.0
inf=SAM3LoRABatchInference(config_path=f"{adl}/configs/full_lora_config.yaml",
   weights_path=f"{adl}/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt",
   checkpoint_path="checkpoints/sam3.pt", resolution=1008, threshold=0.4, device="cuda")
for prompt in ["Gauge_Water","Gauge_Air","water","waterline"]:
    print(f"\n##### prompt='{prompt}' : pred掩码 vs 各GT类别IoU #####")
    accW=[];accA=[]
    for im in coco.loadImgs(coco.getImgIds()):
        p=os.path.join(test_dir,im['file_name'])
        if not os.path.exists(p): continue
        h,w=im['height'],im['width']
        r=inf.process_image(p,[prompt])
        if r is None: print(f"  {im['file_name'][:24]:24s} 未检出"); continue
        pred=np.zeros((h,w),bool)
        for mk in r['masks']: pred|=mk.astype(bool)
        gw=gtm(im['id'],h,w,2); ga=gtm(im['id'],h,w,1)
        iw=iou(pred,gw); ia=iou(pred,ga); accW.append(iw); accA.append(ia)
        cov=pred.sum()/(h*w)
        print(f"  {im['file_name'][:24]:24s} IoU_Water={iw:.2f} IoU_Air={ia:.2f} 覆盖率={cov:.2f} score={r['scores'].max():.2f}")
    if accW: print(f"  >>> 均值 IoU_Water={np.mean(accW):.3f}  IoU_Air={np.mean(accA):.3f}")
