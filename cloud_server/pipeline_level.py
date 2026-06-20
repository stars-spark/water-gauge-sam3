# SAM3 → 水位值 端到端链路打通+验证
# 在9张带真值(文件名L###)的水尺图上：SAM3取水位线像素 → 标定 → 输出水位 → 比对真值
import sys, types, os, re, glob, numpy as np
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
import torch
from PIL import Image
from pycocotools.coco import COCO
from batch_infer_sam_json import SAM3LoRABatchInference

BASE="/media/jiale/AI_Local/水尺水位测量系统"
WL=f"{BASE}/02_数据集/waterline_v1_coco"
ADL=f"{BASE}/03_云端SAM3/autodl_training_backup/SAM3_LoRA-main"

# 1) 收集9张带L真值的图(全划分去重，按原图名)
imgs={}
for split in ["train","valid","test"]:
    for p in glob.glob(f"{WL}/{split}/*.jpg"):
        m=re.search(r'(\d+_[A-Za-z]+(?:_[A-Za-z]+)?_L(\d+))', os.path.basename(p))
        if m:
            key=m.group(1)
            if key not in imgs: imgs[key]=(p, int(m.group(2)), split)
items=sorted(imgs.items(), key=lambda x:x[1][1])
print(f"带真值图 {len(items)} 张：", [(k.split('_L')[0][-3:], v[1]) for k,v in items])

def mask_bottom_top(res, h, w):
    if res is None: return None,None
    pred=np.zeros((h,w),bool)
    for mk in res['masks']: pred|=mk.astype(bool)
    if pred.sum()==0: return None,None
    rows=np.where(pred.any(axis=1))[0]
    return int(rows.max()), int(rows.min())   # bottom, top (像素y，越大越下)

# 2) 两个模型分别跑
def run_model(cfg, weights, prompt):
    inf=SAM3LoRABatchInference(config_path=cfg, weights_path=weights,
        checkpoint_path="checkpoints/sam3.pt", resolution=1008, threshold=0.4, device="cuda")
    out={}
    for key,(p,L,split) in items:
        im=Image.open(p); w,h=im.size
        r=inf.process_image(p,[prompt]); b,t=mask_bottom_top(r,h,w)
        out[key]=(h,b,t)
    del inf; torch.cuda.empty_cache()
    return out

print("\n[1/2] 水尺主体LoRA prompt=water_gauge ...")
gauge=run_model("configs/full_lora_config.yaml","SAM3_LoRa_outputs/best_lora_weights.pt","water_gauge")
print("[2/2] 水位线LoRA prompt=Gauge_Air ...")
air=run_model(f"{ADL}/configs/full_lora_config.yaml",
              f"{ADL}/SAM3_LoRa_Waterline_outputs/best_lora_weights.pt","Gauge_Air")

# 3) GT水位线(标注Gauge_Air下沿)
gtwl={}
for split in ["train","valid","test"]:
    c=COCO(f"{WL}/{split}/_annotations.coco.json")
    fn2id={im['file_name']:im['id'] for im in c.loadImgs(c.getImgIds())}
    for key,(p,L,sp) in items:
        if sp!=split: continue
        iid=fn2id.get(os.path.basename(p));
        if iid is None: continue
        h=c.loadImgs(iid)[0]['height']; w=c.loadImgs(iid)[0]['width']
        m=np.zeros((h,w),bool)
        for a in c.loadAnns(c.getAnnIds(imgIds=iid,catIds=[1])): m|=c.annToMask(a).astype(bool)
        if m.sum()>0: gtwl[key]=int(np.where(m.any(axis=1))[0].max())

# 4) 汇总：各方法的归一化水位线位置 vs L，线性拟合看相关&残差
print(f"\n{'图':18s}{'L真值':>7s}{'H':>6s}{'干区下沿':>9s}{'尺底':>7s}{'GT下沿':>8s}{'干区frac':>9s}")
rows=[]
for key,(p,L,split) in items:
    h,gb,gt_top=gauge[key]; h2,ab,at=air[key]; g=gtwl.get(key)
    air_frac = ab/h if ab else None     # 干区下沿(从顶,越大水越低)... 实际水位 = 1-air_frac 方向待定
    print(f"{key.split('_L')[0][-15:]:18s}{L:7d}{h:6d}{str(ab):>9s}{str(gb):>7s}{str(g):>8s}{(f'{air_frac:.3f}' if air_frac else 'None'):>9s}")
    if ab: rows.append((L, ab/h, (gb/h if gb else np.nan), (g/h if g else np.nan)))

rows=np.array([r for r in rows], float)
def fit(x,y,name):
    ok=~np.isnan(x)&~np.isnan(y)
    if ok.sum()<3: print(f"  {name}: 样本不足"); return
    A=np.polyfit(x[ok],y[ok],1); pred=np.polyval(A,x[ok])
    mae=np.mean(np.abs(pred-y[ok])); r=np.corrcoef(x[ok],y[ok])[0,1]
    print(f"  {name}: L = {A[0]:.1f}*frac + {A[1]:.1f}  | 相关r={r:.3f}  线性拟合后L残差MAE={mae:.1f}")
print("\n线性标定 L(真值) ~ 归一化水位线位置：")
L=rows[:,0]
fit(rows[:,1],L,"干区下沿frac→L ")
fit(rows[:,2],L,"水尺底frac →L ")
fit(rows[:,3],L,"GT下沿frac →L (上界)")
