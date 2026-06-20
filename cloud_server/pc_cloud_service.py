"""
PC 云端 MQTT 服务
版本A（模拟）：收到设备上行数据 → 返回模拟水位，验证全链路通
版本B（真实）：改 USE_REAL_SAM3=True，调 level_pipeline.py 做真实 SAM3 推理

运行：python pc_cloud_service.py
"""
import paho.mqtt.client as mqtt
import json, time, base64, tempfile, os

BROKER       = "broker.emqx.io"   # 换成自建 EMQX IP 后改这里
PORT         = 1883
TOPIC_DATA   = "waterguage/+/data"    # 订阅所有站点上行
TOPIC_RESULT = "waterguage/{sid}/result"
TOPIC_CMD    = "waterguage/{sid}/cmd"

USE_REAL_SAM3 = False   # 改 True 启用真实 SAM3 推理

if USE_REAL_SAM3:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from level_pipeline import WaterLevelMeasurer
    measurer = WaterLevelMeasurer(gauge_top_cm=100.0, gauge_bottom_cm=0.0)
    print("[SAM3] 模型加载完成")


def process(data: dict) -> dict:
    """核心处理：data 为设备上行 JSON，返回水位结果 dict。"""
    sid = data.get("station", "S01")

    # 图像：设备发 image_b64 字段时做真实推理，否则模拟
    image_b64 = data.get("image_b64")

    if USE_REAL_SAM3 and image_b64:
        raw = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(raw); tmp = f.name
        result = measurer.measure(tmp)
        os.unlink(tmp)
        return result

    # 模拟：原样返回设备发来的 level_cm，或给默认值
    level = data.get("level_cm", 50.0)
    return {"ok": True, "level_cm": level, "source": "mock"}


def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] 已连接 {BROKER}，rc={rc}")
    client.subscribe(TOPIC_DATA, qos=0)
    print(f"[MQTT] 订阅 {TOPIC_DATA}")


def on_message(client, userdata, msg):
    topic = msg.topic
    raw   = msg.payload.decode(errors="replace")
    # L610 MQTTPUB 把 AT 命令里的 \" 原样发出，需还原为合法 JSON
    raw_json = raw.replace('\\"', '"')
    print(f"\n[上行] {topic}")
    print(f"  payload: {raw_json[:120]}{'...' if len(raw_json)>120 else ''}")

    try:
        data = json.loads(raw_json)
    except Exception:
        print(f"  [跳过] payload 非 JSON，原始: {raw[:60]}")
        return

    sid = data.get("station", topic.split("/")[1] if "/" in topic else "S01")
    result = process(data)
    print(f"  [解算] {result}")

    # 发布水位结果
    res_topic = TOPIC_RESULT.format(sid=sid)
    client.publish(res_topic, json.dumps({**result, "station": sid,
                                          "ts": int(time.time())}))
    print(f"  [下行] 结果 → {res_topic}")


client = mqtt.Client(client_id="cloud_service_pc")
client.on_connect = on_connect
client.on_message = on_message

print(f"连接 {BROKER}:{PORT} ...")
client.connect(BROKER, PORT, keepalive=60)
client.loop_forever()
