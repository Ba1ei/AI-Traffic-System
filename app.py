import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import base64
from io import BytesIO

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="DRAC Dashboard",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# إزالة هوامش Streamlit
st.markdown("""
<style>
.block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}
header, footer, #MainMenu {
    visibility: hidden;
}
iframe {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Models
# =========================
classification_model = tf.keras.models.load_model("traffic_model.h5")
yolo_model = YOLO("yolov8n.pt")

classes = ["Normal_traffic", "drifting", "wrong_way"]
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

# =========================
# Helper Functions
# =========================
def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def classify_crop(crop_img):
    crop_img = crop_img.convert("RGB").resize((224, 224))
    img_array = np.array(crop_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = classification_model.predict(img_array, verbose=0)[0]
    class_index = int(np.argmax(prediction))

    label = classes[class_index]
    confidence = float(prediction[class_index] * 100)

    return label, confidence, prediction


def detect_and_classify(image):
    image = image.convert("RGB")
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    results = yolo_model(image, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detected_name = yolo_model.names[cls_id]

            if detected_name in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                crop = image.crop((x1, y1, x2, y2))
                label, confidence, prediction = classify_crop(crop)

                is_violation = label != "Normal_traffic"
                color = (239, 68, 68) if is_violation else (34, 197, 94)
                text = f"{label.upper()} {confidence:.1f}%"

                draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
                draw.rectangle((x1, max(0, y1 - 30), x1 + 270, y1), fill=color)
                draw.text((x1 + 8, max(0, y1 - 24)), text, fill="white")

                detections.append({
                    "vehicle": detected_name,
                    "label": label,
                    "confidence": round(confidence, 2),
                    "normal": round(float(prediction[0] * 100), 2),
                    "drifting": round(float(prediction[1] * 100), 2),
                    "wrong_way": round(float(prediction[2] * 100), 2),
                    "violation": "Yes" if is_violation else "No"
                })

    return draw_image, detections


# =========================
# Upload Section
# =========================

st.subheader("Upload or Capture Image")

uploaded_file = st.file_uploader(
    "Upload a traffic image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("Or take a photo")

input_image = None

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)

elif camera_image is not None:
    input_image = Image.open(camera_image)

if input_image is not None:
    image = input_image
    processed_image, detections = detect_and_classify(image)


    total = len(detections)
    violations = sum(1 for d in detections if d["violation"] == "Yes")
    safe = total - violations

    avg_conf = round(
        sum(d["confidence"] for d in detections) / total, 2
    ) if total > 0 else 0

    if total > 0:
        top = max(detections, key=lambda d: d["confidence"])
    else:
        top = {
            "label": "No vehicle",
            "confidence": 0,
            "normal": 0,
            "drifting": 0,
            "wrong_way": 0,
            "vehicle": "-"
        }

    original_b64 = pil_to_base64(image)
    processed_b64 = pil_to_base64(processed_image)

    rows = ""
    for i, d in enumerate(detections, start=1):
        badge_color = "#ef4444" if d["violation"] == "Yes" else "#22c55e"
        result_color = "#f87171" if d["violation"] == "Yes" else "#4ade80"

        rows += f"""
        <tr>
            <td>{i}</td>
            <td>🚗 {d["vehicle"]}</td>
            <td style="color:{result_color}; font-weight:800;">{d["label"]}</td>
            <td>{d["confidence"]}%</td>
            <td><span style="background:{badge_color}; padding:6px 12px; border-radius:8px; font-weight:800;">{d["violation"]}</span></td>
            <td>{d["normal"]}%</td>
            <td>{d["drifting"]}%</td>
            <td>{d["wrong_way"]}%</td>
        </tr>
        """

    if rows == "":
        rows = """
        <tr>
            <td colspan="8">No vehicles detected</td>
        </tr>
        """

    alert_title = "🚨 VIOLATION DETECTED" if violations > 0 else "✅ SAFE TRAFFIC"
    alert_color = "#ef4444" if violations > 0 else "#22c55e"
    behavior = top["label"].replace("_", " ").upper()

    full_dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100vw;
            background: radial-gradient(circle at top left, #071b2d 0%, #020617 45%, #020617 100%);
            color: white;
            font-family: Arial, sans-serif;
            overflow-x: hidden;
        }}

        .container {{
            padding: 28px 34px;
            width: 100%;
            box-sizing: border-box;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }}

        .title {{
            font-size: 48px;
            font-weight: 900;
        }}

        .cyan {{
            color: #22d3ee;
        }}

        .subtitle {{
            color: #cbd5e1;
            font-size: 18px;
            margin-top: 6px;
        }}

        .badge {{
            background: rgba(34,197,94,0.18);
            border: 1px solid #22c55e;
            color: #86efac;
            padding: 14px 24px;
            border-radius: 12px;
            font-weight: 900;
            font-size: 17px;
        }}

        .grid4 {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 18px;
        }}

        .metric {{
            background: rgba(15,23,42,0.95);
            border-radius: 18px;
            padding: 24px;
            display: flex;
            align-items: center;
            gap: 20px;
            min-height: 120px;
        }}

        .blue {{ border: 1px solid #0ea5e9; }}
        .red {{ border: 1px solid #ef4444; }}
        .green {{ border: 1px solid #22c55e; }}
        .purple {{ border: 1px solid #a855f7; }}

        .icon {{
            width: 72px;
            height: 72px;
            border-radius: 50%;
            display: flex;
            justify-content: center;
            align-items: center;
            font-size: 34px;
            border: 4px solid currentColor;
        }}

        .metric-label {{
            font-size: 14px;
            font-weight: 900;
            text-transform: uppercase;
            color: #cbd5e1;
        }}

        .metric-number {{
            font-size: 42px;
            font-weight: 900;
            color: white;
        }}

        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 0.68fr;
            gap: 18px;
            margin-bottom: 18px;
        }}

        .card {{
            background: rgba(15,23,42,0.90);
            border: 1px solid rgba(34,211,238,0.35);
            border-radius: 18px;
            padding: 16px;
        }}

        .card-title {{
            color: #22d3ee;
            font-weight: 900;
            font-size: 18px;
            margin-bottom: 12px;
            text-transform: uppercase;
        }}

        .card img {{
            width: 100%;
            border-radius: 12px;
            display: block;
        }}

        .alert-card {{
            background: rgba(15,23,42,0.95);
            border: 1px solid {alert_color};
            border-radius: 18px;
            padding: 24px;
            text-align: center;
        }}

        .alert-head {{
            background: rgba(239,68,68,0.18);
            color: {alert_color};
            padding: 14px;
            border-radius: 12px;
            font-size: 22px;
            font-weight: 900;
        }}

        .behavior {{
            margin-top: 25px;
            font-size: 26px;
            font-weight: 900;
        }}

        .circle {{
            width: 180px;
            height: 180px;
            border-radius: 50%;
            border: 15px solid {alert_color};
            margin: 28px auto 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }}

        .circle-num {{
            font-size: 34px;
            font-weight: 900;
        }}

        .circle-sub {{
            color: #cbd5e1;
            font-size: 14px;
        }}

        .bottom-grid {{
            display: grid;
            grid-template-columns: 1.35fr 0.95fr;
            gap: 18px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            color: white;
        }}

        th, td {{
            border: 1px solid rgba(148,163,184,0.25);
            padding: 12px;
            text-align: center;
        }}

        th {{
            color: #e5e7eb;
            font-weight: 900;
        }}

        .chart-row {{
            margin-bottom: 22px;
        }}

        .chart-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-weight: 800;
        }}

        .bar-bg {{
            height: 26px;
            background: rgba(148,163,184,0.15);
            border-radius: 8px;
            overflow: hidden;
        }}

        .bar {{
            height: 100%;
            border-radius: 8px;
        }}

        .normal {{ background: #2dd4bf; width: {top["normal"]}%; }}
        .drift {{ background: #ef4444; width: {top["drifting"]}%; }}
        .wrong {{ background: #facc15; width: {top["wrong_way"]}%; }}

        .footer {{
            margin-top: 22px;
            padding-top: 14px;
            border-top: 1px solid rgba(34,211,238,0.25);
            display: flex;
            justify-content: space-between;
            color: #94a3b8;
        }}
    </style>
    </head>

    <body>
    <div class="container">

        <div class="header">
            <div>
                <div class="title">🚁 DRAC <span class="cyan">DASHBOARD</span></div>
                <div class="subtitle">AI Traffic Detection & Behavior Classification</div>
            </div>
            <div class="badge">✓ ANALYSIS COMPLETED</div>
        </div>

        <div class="grid4">
            <div class="metric blue">
                <div class="icon" style="color:#0ea5e9;">🚗</div>
                <div>
                    <div class="metric-label">Vehicles Detected</div>
                    <div class="metric-number">{total}</div>
                </div>
            </div>

            <div class="metric red">
                <div class="icon" style="color:#ef4444;">⚠️</div>
                <div>
                    <div class="metric-label">Violations</div>
                    <div class="metric-number">{violations}</div>
                </div>
            </div>

            <div class="metric green">
                <div class="icon" style="color:#22c55e;">✅</div>
                <div>
                    <div class="metric-label">Safe Vehicles</div>
                    <div class="metric-number">{safe}</div>
                </div>
            </div>

            <div class="metric purple">
                <div class="icon" style="color:#a855f7;">📊</div>
                <div>
                    <div class="metric-label">Avg Confidence</div>
                    <div class="metric-number">{avg_conf}%</div>
                </div>
            </div>
        </div>

        <div class="main-grid">
            <div class="card">
                <div class="card-title">📷 Original Image</div>
                <img src="data:image/png;base64,{original_b64}">
            </div>

            <div class="card">
                <div class="card-title">🎯 Detection Result</div>
                <img src="data:image/png;base64,{processed_b64}">
            </div>

            <div class="alert-card">
                <div class="alert-head">{alert_title}</div>
                <div class="behavior">{behavior}</div>
                <div class="circle">
                    <div class="circle-num">{top["confidence"]}%</div>
                    <div class="circle-sub">Confidence Score</div>
                </div>
            </div>
        </div>

        <div class="bottom-grid">
            <div class="card">
                <div class="card-title">📋 Detection Summary Table</div>
                <table>
                    <tr>
                        <th>#</th>
                        <th>Vehicle</th>
                        <th>Result</th>
                        <th>Confidence</th>
                        <th>Violation</th>
                        <th>Normal</th>
                        <th>Drifting</th>
                        <th>Wrong Way</th>
                    </tr>
                    {rows}
                </table>
            </div>

            <div class="card">
                <div class="card-title">📈 Class Probability Overview</div>

                <div class="chart-row">
                    <div class="chart-label">
                        <span>Normal Traffic</span>
                        <span>{top["normal"]}%</span>
                    </div>
                    <div class="bar-bg"><div class="bar normal"></div></div>
                </div>

                <div class="chart-row">
                    <div class="chart-label">
                        <span>Drifting</span>
                        <span>{top["drifting"]}%</span>
                    </div>
                    <div class="bar-bg"><div class="bar drift"></div></div>
                </div>

                <div class="chart-row">
                    <div class="chart-label">
                        <span>Wrong Way</span>
                        <span>{top["wrong_way"]}%</span>
                    </div>
                    <div class="bar-bg"><div class="bar wrong"></div></div>
                </div>
            </div>
        </div>

        <div class="footer">
            <div>🛡️ DRAC SYSTEM | Intelligent Road Monitoring | Powered by AI</div>
            <div class="cyan">Stay Safe. Drive Smart.</div>
        </div>

    </div>
    </body>
    </html>
    """

    components.html(full_dashboard_html, height=1250, scrolling=True)

else:
    st.info("Upload an image to start the dashboard.")