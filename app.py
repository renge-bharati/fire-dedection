import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="🔥 Fire Detection App",
    page_icon="🔥",
    layout="centered"
)

st.title("🔥 Fire Detection using YOLO")
st.write("Upload an image to detect **fire** using a trained YOLO model.")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # make sure best.pt is in repo root

model = load_model()

# ------------------ IMAGE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("🔍 Detecting fire...")

    # ------------------ YOLO INFERENCE ------------------
    results = model.predict(
        source=img_array,
        conf=0.3,
        save=False
    )

    # Draw results
    annotated_img = img_array.copy()

    fire_detected = False

    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                fire_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"Fire {conf:.2f}"

                cv2.rectangle(
                    annotated_img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    annotated_img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

    # ------------------ SHOW RESULT ------------------
    st.image(
        annotated_img,
        caption="Detection Result",
        use_container_width=True
    )

    if fire_detected:
        st.error("🔥 FIRE DETECTED!")
    else:
        st.success("✅ No fire detected.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("🚀 YOLO Fire Detection | Streamlit App")
