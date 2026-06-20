# -*- coding: utf-8 -*-
"""
设备真实ROI全管线验证：质量门 → SAM3 V2引擎 → E刻度自标定 → 叠加图(目检)
输出 /tmp/device_eval/dev_XX.jpg：绿=水尺掩码 蓝=干区掩码 红线=水位线 + 文本(水位/刻度)
"""
import sys, types, os, glob, re
_m=types.ModuleType('decord'); _m.cpu=lambda *a,**k:None; _m.VideoReader=object
_m.bridge=types.SimpleNamespace(set_bridge=lambda *a,**k:None); sys.modules['decord']=_m
import numpy as np
from PIL import Image, ImageDraw
from measure_engine import WaterLevelMeasurerV2, union_mask, largest_cc
from roi_quality_gate import check_roi
from scale_calibration import calibrate

OUT = "/tmp/device_eval"
os.makedirs(OUT, exist_ok=True)

M = WaterLevelMeasurerV2()
files = sorted(glob.glob("received_images/*.jpg"))
print(f"{'文件':36s}{'门':14s}{'水位线y':>8s}{'conf':>6s}{'cm/px':>8s}{'刻度法':>18s}")
for k, p in enumerate(files):
    fn = os.path.basename(p)
    mconf = re.search(r"conf([0-9]+(?:\.[0-9]+)?)", fn)
    yconf = float(mconf.group(1)) if mconf else None
    im = Image.open(p).convert("RGB"); w, h = im.size

    gate = check_roi(im, yconf)
    if not gate["ok"]:
        print(f"{fn[:36]:36s}REJ:{','.join(gate['reasons'])[:12]:12s}{'—':>8s}")
        continue

    r = M.measure(p)
    if not r.get("ok"):
        print(f"{fn[:36]:36s}{'PASS':14s}{'SAM3未检出':>8s}")
        continue

    gmask = largest_cc(union_mask(M.gauge.process_image(p, ["water_gauge"]), h, w))
    amask = union_mask(M.air.process_image(p, ["Gauge_Air"]), h, w)
    cal = calibrate(im, gmask) if gmask.sum() > 100 else {"ok": False}
    cmpx = f"{cal['cm_per_px']:.3f}" if cal.get("ok") else "—"
    meth = f"{cal.get('method','—')}({cal.get('conf',0):.2f})" if cal.get("ok") else "—"

    ov = np.zeros((h, w, 4), np.uint8)
    ov[gmask] = (0, 230, 0, 80)
    ov[amask] = (40, 90, 255, 70)
    img = im.copy(); img.paste(Image.fromarray(ov), (0, 0), Image.fromarray(ov))
    dr = ImageDraw.Draw(img)
    wl = r["waterline_y"]
    dr.line([(0, wl), (w, wl)], fill=(255, 0, 0), width=max(2, h // 150))
    img = img.resize((w * 3, h * 3), Image.NEAREST)   # 设备图小，放大3x便于目检
    d2 = ImageDraw.Draw(img)
    d2.text((4, 4), f"wl_y={wl:.0f} conf={r['conf']} cm/px={cmpx}", fill=(255, 255, 0))
    img.save(f"{OUT}/dev_{k:02d}.jpg", quality=88)
    print(f"{fn[:36]:36s}{'PASS':14s}{wl:8.0f}{r['conf']:6.2f}{cmpx:>8s}{meth:>18s}")
print("\n叠加图(3x放大)已写入", OUT)
