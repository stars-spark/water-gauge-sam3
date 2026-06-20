# 反光探针：用 water_gauge_v3 中含 reflection 标注的图，验证两件事
#   Q1 干区(Gauge_Air)掩码下沿是否会漏进/越过倒影区 → 反光是否真破坏"干区下沿"水位线法
#   Q2 用 gauge LoRA 的 "reflection" prompt 能否检出倒影掩码(IoU) → 决定反光防御走"减掩码"还是"几何对称截断"
# 用法：PYTHONPATH=. /home/jiale/sam3_test_venv/bin/python probe_reflection.py
import sys, types, os, glob, numpy as np
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
from PIL import Image
from pycocotools.coco import COCO
from measure_engine import WaterLevelMeasurerV2, union_mask, largest_cc

BASE="/media/jiale/AI_Local/水尺水位测量系统"
V3=f"{BASE}/02_数据集/water_gauge_v3_coco"

def iou(a,b):
    a=a.astype(bool); b=b.astype(bool); u=(a|b).sum()
    return float((a&b).sum())/u if u else 0.0

def cat_mask(coco, iid, h, w, cat_ids):
    m=np.zeros((h,w),bool)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, catIds=cat_ids)):
        m|=coco.annToMask(a).astype(bool)
    return m

def bottom(m):
    ys=np.where(m.any(axis=1))[0]; return int(ys.max()) if len(ys) else None
def top(m):
    ys=np.where(m.any(axis=1))[0]; return int(ys.min()) if len(ys) else None

# 收集含 reflection 标注的图
samples=[]
for split in ("train","valid"):
    jp=f"{V3}/{split}/_annotations.coco.json"
    if not os.path.exists(jp): continue
    c=COCO(jp)
    cats={ci['id']:ci['name'] for ci in c.loadCats(c.getCatIds())}
    refl_ids=[i for i,n in cats.items() if 'refl' in n.lower()]
    gauge_ids=[i for i,n in cats.items() if 'gauge' in n.lower()]
    for iid in c.getImgIds():
        if c.getAnnIds(imgIds=iid, catIds=refl_ids):
            info=c.loadImgs(iid)[0]
            samples.append((split, c, iid, info['file_name'], refl_ids, gauge_ids))
print(f"含反光标注图: {len(samples)} 张")

M=WaterLevelMeasurerV2()
print(f"\n{'图':30s}{'尺底':>6s}{'干沿':>6s}{'倒影顶':>7s}{'倒影底':>7s}{'漏入倒影':>9s}{'reflIoU':>8s}")
leaks=[]; refl_ious=[]
for split,c,iid,fn,refl_ids,gauge_ids in samples:
    p=f"{V3}/{split}/{fn}"
    if not os.path.exists(p): continue
    im=Image.open(p); w,h=im.size
    gt_refl=cat_mask(c,iid,h,w,refl_ids)
    rt,rb=top(gt_refl),bottom(gt_refl)
    # 模型预测
    gmask=largest_cc(union_mask(M.gauge.process_image(p,["water_gauge"]),h,w))
    amask=union_mask(M.air.process_image(p,["Gauge_Air"]),h,w)
    refl_pred=union_mask(M.gauge.process_image(p,["reflection"]),h,w)
    g_bot=bottom(gmask); a_bot=bottom(amask)
    refl_iou=iou(refl_pred,gt_refl); refl_ious.append(refl_iou)
    # 漏入倒影 = 干区下沿越过倒影顶多少(正=漏进倒影区)
    leak = (a_bot-rt) if (a_bot is not None and rt is not None) else None
    if leak is not None: leaks.append(leak)
    fmt=lambda v:("%d"%v) if v is not None else "—"
    print(f"{fn[:30]:30s}{fmt(g_bot):>6s}{fmt(a_bot):>6s}{fmt(rt):>7s}{fmt(rb):>7s}{fmt(leak):>9s}{refl_iou:8.2f}")

leaks=np.array([x for x in leaks if x is not None])
print("\n----- 汇总 -----")
if len(leaks):
    print(f"干区下沿 vs 倒影顶 (正=漏进倒影): 中位={np.median(leaks):.0f}px 最大={leaks.max():.0f}px  漏进(>10px)={int((leaks>10).sum())}/{len(leaks)}")
print(f"reflection prompt 检出倒影 IoU: 中位={np.median(refl_ious):.2f} 均值={np.mean(refl_ious):.2f}  可用(>0.3)={int((np.array(refl_ious)>0.3).sum())}/{len(refl_ious)}")
print("\n判读: 漏进>10px多→反光确破坏干区下沿,需防御; reflIoU>0.3可靠→可减倒影掩码,否则走几何对称截断")
