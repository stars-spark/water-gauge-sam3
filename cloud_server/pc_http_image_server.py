"""
PC HTTP 图像接收服务（FastAPI）
-----------------------------------------------
POST /measure  Header: X-Station=S01  Body: JPEG bytes
               -> {"ok": true, "level_cm": 42.3, "station": "S01", "ts": ...}

GET  /health   -> {"status": "ok", "sam3": false}

运行：python pc_http_image_server.py
依赖：pip install fastapi uvicorn pillow

USE_REAL_SAM3=False  模拟模式（现在用）
USE_REAL_SAM3=True   真实SAM3推理（需先建torch+CUDA venv）
"""

import uvicorn, tempfile, os, time, sys, random
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime

USE_REAL_SAM3 = True   # ✅ 2026-06-11 接入V2引擎(measure_engine)：水尺∩干区+沿轴投影+过分割护栏
SAVE_IMAGES   = True   # 保存收到的ROI图像，方便调试

SAVE_DIR = os.path.join(os.path.dirname(__file__), "received_images")
os.makedirs(SAVE_DIR, exist_ok=True)

if USE_REAL_SAM3:
    sys.path.insert(0, os.path.dirname(__file__))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # V2引擎用相对路径加载权重
    from measure_engine import WaterLevelMeasurerV2
    measurer = WaterLevelMeasurerV2(gauge_top_cm=100.0, gauge_bottom_cm=0.0)
    print("[SAM3] V2引擎(水尺∩干区+沿轴投影)加载完成")

app = FastAPI()

# ── 水槽演示：多站时序 + 趋势预测 + 排涝泵自动控制 ──────────────
from collections import defaultdict, deque
from trend_predict import predict_crossing
from pump_controller import PumpController
from temporal_fusion import TemporalFusion

MQTT_BROKER  = "broker.emqx.io"     # 与边缘同一broker；自建EMQX后改这里
MQTT_PORT    = 1883
CONTROL_STATION = "S02"             # 执行闭环的控制对象站(下游)
ALERT_COOLDOWN  = 30.0              # 趋势预警最短发布间隔(秒/站)

HISTORY = defaultdict(lambda: deque(maxlen=64))   # station -> [(ts, level_cm)]  存融合后稳定值
# A3 多帧时域融合：每站一个实例(抗抖动/剔离群/Theil-Sen跟随真实涨落)。水位cm单位→min_jump=2cm
FUSION = defaultdict(lambda: TemporalFusion(window=10, conf_floor=0.3, mad_k=3.5, min_jump=2.0))
_last_alert_ts = {}

_mqtt = None

def _mqtt_pub(topic, payload: dict):
    if _mqtt is None:
        return
    try:
        import json as _json
        _mqtt.publish(topic, _json.dumps(payload))
    except Exception as e:
        print(f"[MQTT发布失败] {topic}: {e}")

def _init_mqtt():
    """连broker、订阅边缘ACK。失败不影响HTTP主流程(泵控指令发不出去而已)。"""
    global _mqtt
    try:
        import paho.mqtt.client as mqtt
        import json as _json
        c = mqtt.Client(client_id=f"cloud-server-{int(time.time())}")
        def _on_msg(cl, ud, msg):
            try:
                if msg.topic == "waterguage/flume/ack":
                    pump_ctl.on_ack(_json.loads(msg.payload.decode()))
                    print(f"[ACK] {msg.payload.decode()[:80]}")
            except Exception as e:
                print(f"[ACK解析失败] {e}")
        c.on_message = _on_msg
        c.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
        c.subscribe("waterguage/flume/ack")
        c.loop_start()
        _mqtt = c
        print(f"[MQTT] 已连 {MQTT_BROKER}:{MQTT_PORT}, 订阅 flume/ack")
    except Exception as e:
        print(f"[MQTT] 不可用({e}), 泵控/预警仅打印不下发")

# 泵控阈值取自 S02.json 的 alarm（on=danger, off=danger-3）
from station_calib import load_station as _load_st
_s02_alarm = (_load_st(CONTROL_STATION) or {}).get("alarm", {})
_danger = _s02_alarm.get("danger_cm", 32.0)
_warn   = _s02_alarm.get("warn_cm", 24.0)
pump_ctl = PumpController(_mqtt_pub, on_cm=_danger, off_cm=_danger - 3.0)
print(f"[泵控] on={_danger} off={_danger-3.0} warn={_warn} (来自stations/{CONTROL_STATION}.json)")


def _flume_logic(station, level_cm, ts, result):
    """每个成功测量样本喂一次：记历史、控泵、趋势预测、发预警。"""
    HISTORY[station].append((ts, level_cm))
    if station != CONTROL_STATION:
        return
    act = pump_ctl.update(station, level_cm, ts)
    if act:
        print(f"[执行闭环] {act} -> waterguage/flume/cmd")
    tr = predict_crossing(list(HISTORY[station]), _warn, now_ts=ts)
    if tr.get("ok"):
        result["forecast"] = tr                      # 回包带预测,边缘屏可显示ETA
        # 上游领涨叙事：把S01斜率一并放进预警
        up = predict_crossing(list(HISTORY["S01"]), _warn, now_ts=ts)
        if tr.get("eta_s") is not None and ts - _last_alert_ts.get(station, 0) >= ALERT_COOLDOWN:
            _last_alert_ts[station] = ts
            alert = {"type": "forecast", "station": station,
                     "eta_s": tr["eta_s"], "slope_cm_per_min": tr["slope_cm_per_min"],
                     "threshold_cm": _warn, "level_now": tr["level_now"], "ts": ts}
            if up.get("ok"):
                alert["upstream_slope_cm_per_min"] = up["slope_cm_per_min"]
            _mqtt_pub("waterguage/alert", alert)
            print(f"[趋势预警] 预计{tr['eta_s']}s后越{_warn}cm (斜率{tr['slope_cm_per_min']}cm/min)")
# ────────────────────────────────────────────────────────────────


def run_inference(jpeg_bytes: bytes) -> dict:
    if USE_REAL_SAM3:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(jpeg_bytes)
            tmp = f.name
        result = measurer.measure(tmp)
        os.unlink(tmp)
        return result
    # 模拟：返回固定偏移水位（接SAM3后删掉）
    return {"ok": True, "level_cm": round(42.0 + random.uniform(-2, 2), 1),
            "source": "mock"}


@app.post("/measure")
async def measure(request: Request):
    body    = await request.body()
    station = request.headers.get("X-Station", "S01")
    conf    = request.headers.get("X-Conf", "0.0")
    print(f"\n[收到图像] station={station}  conf={conf}  size={len(body)/1024:.1f}KB")

    if len(body) < 100:
        return JSONResponse({"ok": False, "msg": "图像数据太小，疑似空包"}, status_code=400)

    # ROI 质量门(毫秒级)：拦掉不可测量的垃圾图，原因码返给设备触发重采
    try:
        import io as _io
        from PIL import Image as _Image
        from roi_quality_gate import check_roi
        gate = check_roi(_Image.open(_io.BytesIO(body)), float(conf))
        if not gate["ok"]:
            print(f"[质量门拦截] reasons={gate['reasons']} metrics={gate['metrics']}")
            return JSONResponse({"ok": False, "msg": "ROI质量不合格",
                                 "reasons": gate["reasons"], "action": "resample",
                                 "station": station, "ts": int(time.time())})
    except Exception as e:
        print(f"[质量门异常,放行] {e}")   # 门坏了不挡主流程

    if SAVE_IMAGES:
        ts_str  = datetime.now().strftime("%H%M%S_%f")[:11]
        fname   = f"{station}_{ts_str}_conf{conf}.jpg"
        fpath   = os.path.join(SAVE_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(body)
        print(f"[保存] {fpath}")

    result = run_inference(body)
    print(f"[解算结果] {result}")

    # ★输出合法性门(2026-06-20)：引擎判读数越界(水位线落在水尺外=gauge主体检测碎片化)→不入库不融合,
    #   直接触发重采,避免"读错还上报"污染时域融合/泵控。批量验证拦7/9灾难且0误杀正常图。
    if result.get("ok") and result.get("reliable") is False:
        print(f"[合法性门拦截] {result.get('reason')}")
        return JSONResponse({"ok": False, "msg": "读数越界,疑水尺主体检测碎片化",
                             "reason": result.get("reason"), "action": "resample",
                             "station": station, "ts": int(time.time())})

    # 按站标定 → 绝对水位（设备需带 X-ROI 头="x,y,w,h"，即YOLO裁剪框在全幅中的位置）
    if result.get("ok") and result.get("waterline_y") is not None:
        from station_calib import load_station, parse_roi_header, absolute_level
        cfg = load_station(station)
        roi = parse_roi_header(request.headers.get("X-ROI", ""))
        ab = absolute_level(result["waterline_y"], roi, cfg)
        if ab.get("ok"):
            result["level_cm"] = ab["level_cm"]          # 绝对水位覆盖相对值
            result["calib_method"] = ab["method"]
            alarm = (cfg or {}).get("alarm", {})
            if alarm.get("danger_cm") is not None and ab["level_cm"] >= alarm["danger_cm"]:
                result["alarm"] = "danger"
            elif alarm.get("warn_cm") is not None and ab["level_cm"] >= alarm["warn_cm"]:
                result["alarm"] = "warn"
            print(f"[绝对水位] {ab['level_cm']}cm ({ab['method']}) alarm={result.get('alarm','-')}")
        else:
            result["calib_method"] = "relative_gauge_range"   # 无站配置→保持V2相对值
            print(f"[标定] 用相对值({ab.get('msg')})")

    # A3 多帧时域融合：逐帧水位→稳定读数(抗抖动/剔离群/跟随真实涨落)。下游(入库/泵控/趋势)统一用融合值,
    #   避免单帧抽风(残留反光帧等)误触发开泵/误报趋势。被剔除帧仍返回当前稳定估计。
    if result.get("ok") and isinstance(result.get("level_cm"), (int, float)):
        fr = FUSION[station].update(float(result["level_cm"]), conf=result.get("conf", 1.0))
        result["level_cm_raw"] = result["level_cm"]
        if fr["fused"] is not None:
            result["level_cm"] = fr["fused"]                      # 融合值覆盖,供显示+下游
        result["fusion"] = {k: fr[k] for k in ("accepted", "reason", "fused", "n_window", "spread", "stable")}
        if not fr["accepted"]:
            print(f"[时域融合] 本帧{fr['reason']}: raw={result['level_cm_raw']} -> 稳定值={fr['fused']}")

    # 水槽演示逻辑：时序入库 → 泵控 → 趋势预测/预警（用融合后的稳定水位）
    if result.get("ok") and isinstance(result.get("level_cm"), (int, float)):
        try:
            _flume_logic(station, float(result["level_cm"]), int(time.time()), result)
        except Exception as e:
            print(f"[水槽逻辑异常,不影响回包] {e}")

    return JSONResponse({
        **result,
        "station": station,
        "ts": int(time.time())
    })


@app.get("/health")
async def health():
    return {"status": "ok", "sam3": USE_REAL_SAM3, "ts": int(time.time())}


@app.get("/history")
async def history(station: str = "S02", n: int = 32):
    pts = list(HISTORY.get(station, []))[-n:]
    return {"station": station,
            "points": [{"ts": ts, "level_cm": lv} for ts, lv in pts]}


if __name__ == "__main__":
    _init_mqtt()
    local_ip = "0.0.0.0"
    port     = 8000
    print("=" * 50)
    print(f"HTTP 图像服务启动  http://{local_ip}:{port}")
    print(f"SAM3模式: {'真实RTX4080' if USE_REAL_SAM3 else '模拟'}")
    print("⚠  请把你的PC局域网IP告诉设备脚本（SERVER_IP常量）")
    print("=" * 50)
    uvicorn.run(app, host=local_ip, port=port, log_level="warning")
