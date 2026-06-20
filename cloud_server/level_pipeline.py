# 水位测量端到端模块：图像 → 水位值(cm)
# 链路：SAM3水尺LoRA(定尺+尺度) + SAM3干区LoRA(水位线像素) → 线性标定 → 水位读数
import sys, types, os, re, glob, numpy as np
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
import torch
from PIL import Image
from pycocotools.coco import COCO
from batch_infer_sam_json import SAM3LoRABatchInference

import platform
BASE = (r"D:\水尺水位测量系统" if platform.system()=="Windows"
        else "/media/jiale/AI_Local/水尺水位测量系统")
WL  = os.path.join(BASE, "02_数据集", "waterline_v1_coco")
ADL = os.path.join(BASE, "03_云端SAM3", "autodl_training_backup", "SAM3_LoRA-main")

class WaterLevelMeasurer:
    """固定站水位测量：水尺物理量程已知(reading_top..reading_bottom，单位cm)。
       水位 = reading_bottom + (y_bottom - y_waterline)/(y_bottom - y_top)*(reading_top-reading_bottom)"""
    def __init__(self, gauge_top_cm=100.0, gauge_bottom_cm=0.0):
        self.rt, self.rb = gauge_top_cm, gauge_bottom_cm
        print("加载 水尺主体LoRA ..."); self.gauge=SAM3LoRABatchInference(
            "configs/full_lora_config.yaml","SAM3_LoRa_outputs/best_lora_weights.pt",
            "checkpoints/sam3.pt", 1008, 0.4, "cuda")
        print("加载 干区(水位线)LoRA ..."); self.air=SAM3LoRABatchInference(
            f"{ADL}/configs/full_lora_config.yaml",
            f"{ADL}/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt",
            "checkpoints/sam3.pt", 1008, 0.4, "cuda")

    @staticmethod
    def _extent(res,h,w):
        if res is None: return None
        m=np.zeros((h,w),bool)
        for mk in res['masks']: m|=mk.astype(bool)
        if m.sum()==0: return None
        r=np.where(m.any(axis=1))[0]; return int(r.min()),int(r.max())  # top,bottom

    def measure(self, image_path):
        im=Image.open(image_path); w,h=im.size
        ge=self._extent(self.gauge.process_image(image_path,["water_gauge"]),h,w)
        ae=self._extent(self.air.process_image(image_path,["Gauge_Air"]),h,w)
        if ge is None: return {"ok":False,"msg":"未检出水尺"}
        y_top,y_bot=ge
        y_wl = ae[1] if ae else y_bot          # 水位线=干区下沿；缺失则退化为尺底
        gpx = max(1, y_bot-y_top)
        level = self.rb + (y_bot - y_wl)/gpx*(self.rt-self.rb)
        cm_per_px=(self.rt-self.rb)/gpx
        return {"ok":True,"level_cm":round(level,1),"y_waterline":y_wl,
                "gauge_top":y_top,"gauge_bottom":y_bot,"cm_per_px":round(cm_per_px,4)}

if __name__=="__main__":
    M=WaterLevelMeasurer(gauge_top_cm=100.0, gauge_bottom_cm=0.0)
    # 真实照片(SAM3擅长) vs 合成图(域差)，并用GT水位线算端到端cm误差
    reals=sorted(glob.glob(f"{WL}/test/water_gauge_*.jpg"))
    print(f"\n=== A) 真实照片端到端水位输出({len(reals)}张) ===")
    # GT水位线
    c=COCO(f"{WL}/test/_annotations.coco.json"); fn2id={im['file_name']:im['id'] for im in c.loadImgs(c.getImgIds())}
    def gtwl(fn,h,w):
        iid=fn2id.get(fn); m=np.zeros((h,w),bool)
        for a in c.loadAnns(c.getAnnIds(imgIds=iid,catIds=[1])): m|=c.annToMask(a).astype(bool)
        return int(np.where(m.any(axis=1))[0].max()) if m.sum()>0 else None
    errs=[]
    for p in reals:
        r=M.measure(p)
        if not r["ok"]: print(f"  {os.path.basename(p)[:24]} {r['msg']}"); continue
        im=Image.open(p); w,h=im.size; g=gtwl(os.path.basename(p),h,w)
        gpx=r["gauge_bottom"]-r["gauge_top"]
        if g and gpx>0:
            gt_level=(r["gauge_bottom"]-g)/gpx*100.0; err=abs(r["level_cm"]-gt_level); errs.append(err)
            print(f"  {os.path.basename(p)[:22]:22s} 输出水位={r['level_cm']:5.1f}cm  GT={gt_level:5.1f}cm  误差={err:4.1f}cm")
    if errs:
        print(f"\n>>> 真实照片端到端水位 MAE={np.mean(errs):.1f}cm  中位={np.median(errs):.1f}cm (n={len(errs)})")
        print(f">>> (按标准1米水尺；误差主要来自SAM3水位线像素，换算本身精确)")
