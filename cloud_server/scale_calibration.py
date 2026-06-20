# -*- coding: utf-8 -*-
"""
E字刻度尺度自标定：水尺图 + 水尺掩码 → cm/px（不依赖整尺可见）
原理：标准水尺 E 字刻度每个 E 高 5cm。沿水尺主轴取强度剖面，
自相关找主周期 P(px) → cm_per_px = 5.0 / P。
只需局部几个 E 可见即可，天然适配被裁切/半泡水的 ROI。
依赖：numpy + PIL。掩码可来自 GT 标注(开发期)或 SAM3 推理(上线期)。
"""
import numpy as np
from PIL import Image

E_CM = 5.0          # 一个 E 字的物理高度
LR_CM = 10.0        # E字左右交替的周期(左E 5cm + 右E 5cm)
MIN_PERIOD_PX = 6   # 周期下限(px)，低于此认为是噪声/锯齿
MIN_E_COUNT = 3     # 掩码长度至少容纳几个周期才可信


def largest_instance(mask):
    """多实例(多根水尺/商品图多块板)时取最大连通域——标定必须按单实例做，
    否则 PCA 主轴会被实例排布方向带偏(实测：三块板平铺图主轴变成水平)。"""
    from scipy import ndimage
    m = mask.astype(bool)
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def gauge_axis(mask):
    """PCA 求水尺主轴。返回 (center(2,), v(2,) 单位向量[沿尺向下为正], t值数组, mask像素坐标)。"""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    d = pts - c
    cov = d.T @ d / len(d)
    w, V = np.linalg.eigh(cov)
    v = V[:, np.argmax(w)]          # 主方向
    if v[1] < 0:                    # 统一为图像 y 增大方向(向下)
        v = -v
    t = d @ v                       # 各像素沿轴参数
    return c, v, t, pts


def _binned_profile(vals, idx, nbin):
    s = np.bincount(idx, weights=vals, minlength=nbin)
    n = np.bincount(idx, minlength=nbin)
    prof = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    ok = ~np.isnan(prof)
    if ok.sum() < 2:
        return np.zeros(nbin)
    prof = np.interp(np.arange(nbin), np.nonzero(ok)[0], prof[ok])
    win = max(15, nbin // 4) | 1
    kern = np.ones(win) / win
    trend = np.convolve(np.pad(prof, win // 2, mode="edge"), kern, "valid")
    return prof - trend


def axis_profile(img_gray, mask, c, v, t):
    """沿主轴 1px 步长的强度剖面（掩码内横截面均值，去趋势）。"""
    t0 = t.min()
    nbin = int(t.max() - t0) + 1
    idx = np.clip((t - t0).astype(int), 0, nbin - 1)
    ys, xs = np.nonzero(mask)
    return np.arange(nbin) + t0, _binned_profile(img_gray[ys, xs].astype(np.float64), idx, nbin)


def lr_diff_profile(img_gray, mask, c, v, t):
    """左右半区差分剖面：标准水尺 E 字左右交替，单侧'有E/无E'交替周期=10cm。
    差分信号锁定 10cm 周期，天然消掉 E 内部横条(2cm)与 5cm 的谐波歧义。"""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    u = np.array([-v[1], v[0]])                   # 垂直于主轴
    side = (pts - c) @ u
    t0 = t.min()
    nbin = int(t.max() - t0) + 1
    idx = np.clip((t - t0).astype(int), 0, nbin - 1)
    g = img_gray[ys, xs].astype(np.float64)
    L = _binned_profile(g[side < 0], idx[side < 0], nbin)
    R = _binned_profile(g[side >= 0], idx[side >= 0], nbin)
    return L - R


def _autocorr(x):
    x = x - x.mean()
    n = len(x)
    ac = np.correlate(x, x, "full")[n - 1:]
    return ac / (ac[0] + 1e-9)


def _refine(ac, i):
    """峰值亚像素二次插值。"""
    if 1 <= i < len(ac) - 1:
        denom = ac[i - 1] - 2 * ac[i] + ac[i + 1]
        if abs(denom) > 1e-9:
            return i + 0.5 * (ac[i - 1] - ac[i + 1]) / denom
    return float(i)


def dominant_period(detrended, antiphase=False):
    """自相关找主周期(px)。antiphase=True 时要求半周期处为负谷
    （左右交替的10cm方波特征：ac(P/2)<0；2cm横条冒充不了），并按谷深加权置信度。
    返回 (period, 置信度0~1)；失败 (None, 0)。"""
    n = len(detrended)
    if n < MIN_PERIOD_PX * MIN_E_COUNT:
        return None, 0.0
    ac = _autocorr(detrended)
    lo, hi = MIN_PERIOD_PX, n // MIN_E_COUNT
    if hi <= lo:
        return None, 0.0
    seg = ac[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]
             and seg[i] > 0.1]
    if not peaks:
        return None, 0.0
    if not antiphase:
        best = max(peaks, key=lambda i: seg[i])
        return _refine(ac, lo + best), float(np.clip(seg[best], 0, 1))
    # 反相校验：在所有候选峰里找满足 ac(P/2)<-0.05 的、谷最深者
    best_p, best_score = None, 0.0
    for i in peaks:
        P = lo + i
        half = P // 2
        if half < 2 or half >= len(ac):
            continue
        dip = float(ac[half])
        if dip < -0.05:
            score = float(np.clip(seg[i], 0, 1)) * float(np.clip(-dip, 0, 1)) ** 0.5
            if score > best_score:
                best_p, best_score = P, score
    if best_p is None:
        return None, 0.0
    return _refine(ac, best_p), best_score


def calibrate(img, mask, single_instance=True):
    """主入口。img: PIL.Image 或 ndarray；mask: bool ndarray(单实例；多实例自动取最大连通域)。
    先用左右差分剖面锁 10cm 周期(抗谐波)；失败回退灰度剖面按 5cm 解释。
    返回 dict(cm_per_px, period_px, period_cm, conf, method, n_periods)。失败 ok=False。"""
    if isinstance(img, Image.Image):
        g = np.asarray(img.convert("L"), np.float64)
    else:
        a = np.asarray(img)
        g = a.mean(2) if a.ndim == 3 else a.astype(np.float64)
    mask = mask.astype(bool)
    if single_instance:
        mask = largest_instance(mask)
    if mask.sum() < 100:
        return {"ok": False, "msg": "掩码太小"}
    c, v, t, _ = gauge_axis(mask)

    # 主路线：左右差分 + 反相校验 → 10cm 周期(半周期处必须负谷,2cm横条冒充不了)
    diff = lr_diff_profile(g, mask, c, v, t)
    p_lr, conf_lr = dominant_period(diff, antiphase=True)
    # 回退：灰度剖面 → 按 5cm 解释(存在2/5cm谐波歧义,置信度折半)
    _, det = axis_profile(g, mask, c, v, t)
    p_g, conf_g = dominant_period(det)

    if p_lr is not None and conf_lr >= 0.15:
        period, conf, cm, method = p_lr, conf_lr, LR_CM, "lr_diff_10cm"
    elif p_g is not None:
        period, conf, cm, method = p_g, conf_g * 0.5, E_CM, "gray_5cm_ambiguous"
    else:
        return {"ok": False, "msg": "未找到刻度周期"}
    n_per = (t.max() - t.min()) / period
    implied_len = float(n_per * cm)                         # 隐含可见尺长(cm)=周期数×周期cm
    # ★置信门(2026-06-19实测加)：谐波锁错时周期偏小→隐含尺长荒谬(实测000150=399cm=4m尺,物理不可能)。
    #   单根水尺合理尺长~25~320cm;超界或conf过低→标 reliable=False,调用方应弃用E、回退两点标定。
    reliable = bool(25.0 <= implied_len <= 320.0 and conf >= 0.15)
    return {"ok": True, "cm_per_px": cm / period, "period_px": period, "period_cm": cm,
            "conf": round(conf, 3), "method": method, "center": c.tolist(),
            "axis_v": v.tolist(), "n_periods": round(float(n_per), 1),
            "implied_len_cm": round(implied_len, 1), "reliable": reliable}


def render_ticks(img, mask, calib, out_path):
    """可视化：主轴线 + 每个周期一道垂直刻度线，供目检对齐 E 字边界。"""
    from PIL import ImageDraw
    im = img.convert("RGB").copy() if isinstance(img, Image.Image) else Image.fromarray(img).convert("RGB")
    dr = ImageDraw.Draw(im)
    c = np.array(calib["center"]); v = np.array(calib["axis_v"])
    u = np.array([-v[1], v[0]])                   # 垂直方向
    ys, xs = np.nonzero(mask)
    t = (np.stack([xs, ys], 1) - c) @ v
    t0, t1 = t.min(), t.max()
    p0, p1 = c + v * t0, c + v * t1
    dr.line([tuple(p0), tuple(p1)], fill=(0, 160, 255), width=2)
    P = calib["period_px"]
    half = max(6, int(0.6 * (xs.max() - xs.min()) / 2))
    k = 0
    tt = t0
    while tt <= t1:
        q = c + v * tt
        a, b = q - u * half, q + u * half
        dr.line([tuple(a), tuple(b)], fill=(255, 40, 40), width=2)
        k += 1
        tt = t0 + k * P
    im.save(out_path, quality=90)
    return out_path
