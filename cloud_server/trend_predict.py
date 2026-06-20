# -*- coding: utf-8 -*-
"""水位趋势预测：对单站时序做最小二乘线性外推，给出"预计越线时间"。

设计约束：
- 演示水槽 60~120s 涨满，窗口默认取最近 8 点 / 5 分钟；真实场景把
  min_pts/max_age_s 调大即可，接口不变。
- 纯 numpy，无额外依赖；输入输出都是朴素 dict，便于直接塞进 HTTP 回包
  和 MQTT alert payload。
"""
import time


def fit_trend(history):
    """对 [(ts, level_cm), ...]（时间升序）做最小二乘直线拟合。

    返回 (intercept, slope_cm_per_s, r2)；点数<3 或时间跨度为0 返回 None。
    intercept 是 t=history[0].ts 处的水位（把时间原点移到首点，避免大时间戳
    数值病态）。
    """
    if len(history) < 3:
        return None
    t0 = history[0][0]
    xs = [ts - t0 for ts, _ in history]
    ys = [lv for _, lv in history]
    n = len(xs)
    if xs[-1] - xs[0] <= 0:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    syy = sum((y - mean_y) ** 2 for y in ys)
    if syy == 0:
        r2 = 1.0          # 完全水平的序列：拟合精确，但 slope=0 不会预报越线
    else:
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 - ss_res / syy
    return intercept, slope, r2


def predict_crossing(history, threshold_cm, now_ts=None,
                     min_pts=4, max_age_s=600, min_slope=1e-4):
    """线性外推预计越线时间。

    history      [(ts, level_cm), ...] 时间升序（deque 转 list 传入）
    threshold_cm 预警阈值（一般传站点 warn_cm）
    返回 dict：
      ok=True:  {"ok":True, "slope_cm_per_min":x, "r2":x,
                 "eta_s":float|None, "crossing_ts":int|None,
                 "level_now":x}
                eta_s=None → 斜率太小不会越线；eta_s=0 → 已在阈值上方
      ok=False: {"ok":False, "msg":...}  点数不足/数据过旧/拟合失败
    """
    now_ts = now_ts if now_ts is not None else int(time.time())
    pts = [(ts, lv) for ts, lv in history if now_ts - ts <= max_age_s]
    if len(pts) < min_pts:
        return {"ok": False, "msg": "样本不足(%d<%d)" % (len(pts), min_pts)}
    fit = fit_trend(pts)
    if fit is None:
        return {"ok": False, "msg": "拟合失败(时间跨度/方差为0)"}
    intercept, slope, r2 = fit
    t0 = pts[0][0]
    level_now = intercept + slope * (now_ts - t0)
    out = {"ok": True,
           "slope_cm_per_min": round(slope * 60.0, 3),
           "r2": round(r2, 3),
           "level_now": round(level_now, 2),
           "threshold_cm": threshold_cm}
    if level_now >= threshold_cm:
        out["eta_s"] = 0.0
        out["crossing_ts"] = now_ts
    elif slope <= min_slope:
        out["eta_s"] = None          # 不涨或在降，不会越线
        out["crossing_ts"] = None
    else:
        eta = (threshold_cm - level_now) / slope
        out["eta_s"] = round(eta, 1)
        out["crossing_ts"] = int(now_ts + eta)
    return out


if __name__ == "__main__":
    # 自测：合成 0.05cm/s 斜坡 + 噪声，验证 eta 误差 < 5%
    import random
    random.seed(42)
    t0 = 1_700_000_000
    slope_true = 0.05                       # cm/s = 3 cm/min
    hist = [(t0 + i * 5, 10.0 + slope_true * i * 5 + random.uniform(-0.15, 0.15))
            for i in range(8)]              # 35s 历史，水位 10→11.75
    now = t0 + 35
    thr = 16.0                              # 还差 ~4.25cm → 理论 eta ≈ 85s
    r = predict_crossing(hist, thr, now_ts=now)
    eta_true = (thr - (10.0 + slope_true * 35)) / slope_true
    print("预测:", r)
    print("理论 eta=%.1fs  误差=%.1f%%" % (eta_true, abs(r["eta_s"] - eta_true) / eta_true * 100))
    assert r["ok"] and abs(r["eta_s"] - eta_true) / eta_true < 0.05

    # 边界：水平序列不报、已越线 eta=0、样本不足
    flat = [(t0 + i * 5, 12.0) for i in range(8)]
    assert predict_crossing(flat, 16.0, now_ts=now)["eta_s"] is None
    high = [(t0 + i * 5, 17.0 + 0.05 * i) for i in range(8)]
    assert predict_crossing(high, 16.0, now_ts=now)["eta_s"] == 0.0
    assert not predict_crossing(hist[:2], 16.0, now_ts=now)["ok"]
    print("自测全部通过 ✓")
