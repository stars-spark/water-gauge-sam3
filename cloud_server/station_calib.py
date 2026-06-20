# -*- coding: utf-8 -*-
"""
按站标定与像素→绝对水位换算（任务A1第一步：一次性几何标定，固定站长期有效）
=====================================================================
坐标系约定（关键）：
  锚点/两点标定的 y_px 均为【整幅相机画面】坐标(y向下)。设备上传的是 ROI 裁剪图，
  必须随包带 ROI 偏移(X-ROI: "x,y,w,h")，云端用 y_full = y_roi + roi_y 换回全幅坐标。
  ——相机和水尺都固定(B1刚性安装)，全幅坐标系才是长期稳定的参考系。

三种标定模式(按精度排序，配置里 mode 指定)：
  1. two_point   安装时人工标两处已知读数的像素行(如 80cm 刻度线在 y=312, 30cm 在 y=1156)
                 → 同时解出 cm_per_px 和锚点。最准，推荐。
  2. anchor_scale 一个锚点(y_px↔reading_cm) + cm_per_px
                 (cm_per_px 可填实测值；缺省时用 E 字刻度自标定的输出)
  3. gauge_range  整尺可见假设(尺顶=top_cm,尺底=bottom_cm)——V2引擎原行为，兜底用。

换算(y 向下增大、读数向上增大)：
  level_cm = reading_ref - (y_wl_full - y_ref) * cm_per_px
=====================================================================
"""
import json, os

STATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stations")


def load_station(station_id):
    """读取站点配置；不存在返回 None（调用方回退 gauge_range 模式）。"""
    p = os.path.join(STATIONS_DIR, f"{station_id}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def parse_roi_header(s):
    """解析 X-ROI 头 'x,y,w,h' → dict；非法返回 None。"""
    try:
        x, y, w, h = [int(v) for v in s.strip().split(",")]
        return {"x": x, "y": y, "w": w, "h": h}
    except Exception:
        return None


def absolute_level(y_wl_roi, roi, cfg, auto_cm_per_px=None):
    """ROI内水位线行 → 绝对水位cm。
    y_wl_roi: V2引擎输出的 waterline_y(ROI坐标)
    roi:      parse_roi_header 的结果(None=图即全幅)
    cfg:      load_station 的配置(None=无法换算)
    auto_cm_per_px: E字刻度自标定输出(anchor_scale 模式缺 cm_per_px 时启用)
    返回 {"ok","level_cm","method","cm_per_px"} 或 {"ok":False,"msg"}"""
    if cfg is None:
        return {"ok": False, "msg": "无站点标定配置"}
    calib = cfg.get("calib", {})
    mode = calib.get("mode")
    y_full = float(y_wl_roi) + (roi["y"] if roi else 0)

    if mode == "two_point":
        tp = calib.get("two_point", {})
        p1, p2 = tp.get("p1"), tp.get("p2")
        if not (p1 and p2) or p1["y_px"] == p2["y_px"]:
            return {"ok": False, "msg": "two_point 配置不完整"}
        cm_per_px = (p1["reading_cm"] - p2["reading_cm"]) / (p2["y_px"] - p1["y_px"])
        if cm_per_px <= 0:
            return {"ok": False, "msg": "two_point 两点关系不自洽(读数应随y增大而减小)"}
        level = p1["reading_cm"] - (y_full - p1["y_px"]) * cm_per_px
        return {"ok": True, "level_cm": round(level, 1), "method": "two_point",
                "cm_per_px": round(cm_per_px, 4)}

    if mode == "anchor_scale":
        an = calib.get("anchor")
        if not an:
            return {"ok": False, "msg": "anchor 缺失"}
        cm_per_px = calib.get("cm_per_px") or auto_cm_per_px
        if not cm_per_px:
            return {"ok": False, "msg": "无 cm_per_px(配置未填且自动刻度不可用)"}
        src = "cfg" if calib.get("cm_per_px") else "auto_E"
        level = an["reading_cm"] - (y_full - an["y_px"]) * cm_per_px
        return {"ok": True, "level_cm": round(level, 1),
                "method": f"anchor_scale({src})", "cm_per_px": round(cm_per_px, 4)}

    return {"ok": False, "msg": f"未知标定模式: {mode}"}


# ===================== 自测（纯算术，不依赖模型）=====================
if __name__ == "__main__":
    # 模拟站：全幅里 80cm 刻度在 y=312、30cm 在 y=1156 → cm_per_px=50/844≈0.05924
    cfg = {"calib": {"mode": "two_point",
                     "two_point": {"p1": {"y_px": 312, "reading_cm": 80.0},
                                   "p2": {"y_px": 1156, "reading_cm": 30.0}}}}
    roi = {"x": 100, "y": 250, "w": 200, "h": 1100}      # 设备裁剪框
    # 设水位线在全幅 y=900 → 期望 80 - (900-312)*0.05924 ≈ 45.2cm
    r = absolute_level(900 - roi["y"], roi, cfg)
    print("two_point:", r, "(期望≈45.2)")
    assert r["ok"] and abs(r["level_cm"] - 45.2) < 0.2

    cfg2 = {"calib": {"mode": "anchor_scale",
                      "anchor": {"y_px": 312, "reading_cm": 80.0}}}
    r2 = absolute_level(900 - roi["y"], roi, cfg2, auto_cm_per_px=0.05924)
    print("anchor+autoE:", r2, "(期望≈45.2)")
    assert r2["ok"] and abs(r2["level_cm"] - 45.2) < 0.2

    r3 = absolute_level(650, None, cfg)                   # 无ROI(整幅直传)
    print("整幅无ROI:", r3, "(期望=60.0)")
    assert r3["ok"] and abs(r3["level_cm"] - 60.0) < 0.1
    print("✅ station_calib 自测全部通过")
