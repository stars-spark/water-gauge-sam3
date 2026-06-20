# -*- coding: utf-8 -*-
"""
ROI 质量门：在跑 SAM3 之前用毫秒级检查拦掉不可测量的垃圾 ROI。
设计哲学(实测调出来的)：**只拦确定的垃圾，存疑放行**——下游 SAM3 本身是强过滤器
(检不出水尺会返回失败)，门的价值是省 0.5s GPU + 给边缘早期"重采"反馈。
⚠️ 不做 blurry/低对比度硬阈值：实测会误杀 SAM3 能测到 6px 精度的"软图"。

判据(2026-06-11 在设备真实ROI 12张 + WaterLine test 16张上验证：垃圾2/2拦截,好图26/26放行)：
  - 尺寸/面积/YOLO置信度 下限
  - 长宽比 < 1.3(水尺ROI天然细长)
  - no_structure = 周期置信<0.10 且 长轴过零<40 —— 纯色块/水尺一角的铁证组合
    (真水尺即使低分辨率周期峰被稀释,交替结构的过零数仍然高)
依赖：numpy + PIL + scale_calibration.dominant_period
用法：
    from roi_quality_gate import check_roi
    r = check_roi(pil_img, yolo_conf=0.74)
    # -> {"ok":bool, "reasons":[...], "metrics":{...}}  reasons空=通过
    # 拒绝时建议云端向边缘下发 resample 指令(原因码随包带回)
"""
import numpy as np
from PIL import Image
from scale_calibration import dominant_period

MIN_SIDE_PX  = 16     # 最短边下限
MIN_AREA_PX  = 2000   # 总像素下限
MIN_CONF     = 0.45   # YOLO置信度下限(云端复核)
MIN_ASPECT   = 1.3    # 长边/短边下限
PCONF_FLOOR  = 0.10   # 周期置信下限 } 两者同时低
ZC_FLOOR     = 40     # 长轴过零下限 } 才判垃圾


def _axis_features(g):
    """长轴均值剖面 → (去趋势剖面, 过零次数)。"""
    h, w = g.shape
    prof = g.mean(axis=1) if h >= w else g.mean(axis=0)
    n = len(prof)
    win = max(9, n // 6) | 1
    trend = np.convolve(np.pad(prof, win // 2, mode="edge"), np.ones(win) / win, "valid")
    det = prof - trend
    zc = int(np.sum(np.diff(np.sign(det - det.mean())) != 0))
    return det, zc


def check_roi(img, yolo_conf=None):
    """img: PIL.Image / ndarray。返回 {"ok","reasons","metrics"}。"""
    if isinstance(img, Image.Image):
        g = np.asarray(img.convert("L"), np.float64)
    else:
        a = np.asarray(img)
        g = a.mean(2) if a.ndim == 3 else a.astype(np.float64)
    h, w = g.shape
    aspect = max(h, w) / max(1, min(h, w))
    det, zc = _axis_features(g)
    _, pconf = dominant_period(det)

    reasons = []
    if min(h, w) < MIN_SIDE_PX:
        reasons.append("too_small_side")
    if h * w < MIN_AREA_PX:
        reasons.append("too_small_area")
    if yolo_conf is not None and yolo_conf < MIN_CONF:
        reasons.append("low_conf")
    if aspect < MIN_ASPECT:
        reasons.append("bad_aspect")
    if pconf < PCONF_FLOOR and zc < ZC_FLOOR:
        reasons.append("no_structure")     # 纯色块/天空/水尺一角

    return {"ok": not reasons, "reasons": reasons,
            "metrics": {"w": w, "h": h, "aspect": round(aspect, 2),
                        "period_conf": round(float(pconf), 3), "zero_cross": zc,
                        "conf": yolo_conf}}


if __name__ == "__main__":
    import glob, os, re, sys
    folders = sys.argv[1:] or ["received_images"]
    for fo in folders:
        n_pass = n_all = 0
        print(f"\n== {fo} ==")
        for p in sorted(glob.glob(os.path.join(fo, "*.jpg")))[:40]:
            fn = os.path.basename(p)
            m = re.search(r"conf([0-9]+(?:\.[0-9]+)?)", fn)
            conf = float(m.group(1)) if m else None
            r = check_roi(Image.open(p), conf)
            mt = r["metrics"]
            n_all += 1; n_pass += r["ok"]
            print(f"  {fn[:40]:40s} {'PASS' if r['ok'] else 'REJ:'+','.join(r['reasons']):24s}"
                  f" pconf={mt['period_conf']:.2f} zc={mt['zero_cross']:3d} aspect={mt['aspect']}")
        print(f"  -> {n_pass}/{n_all} PASS")
