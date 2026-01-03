# =========================
# Imports
# =========================
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import io
import time
from PIL import Image
from torchvision import transforms
from datetime import datetime
import matplotlib.pyplot as plt

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Crop Disease Detection with Explainable AI",
    page_icon="🌿",
    layout="wide"
)

# =========================
# Custom Styling (DARK MODE SAFE)
# =========================
st.markdown("""
<style>
.title {
    font-size:42px;
    font-weight:700;
}
.subtitle {
    font-size:18px;
    color:#9aa0a6;
}
.small {
    color:#9aa0a6;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🌿 Crop Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning + Explainable AI (Grad-CAM)</div>', unsafe_allow_html=True)
st.markdown("---")

# =========================
# Device
# =========================
device = "cpu"

# =========================
# Class Labels (EXACT ORDER)
# =========================
CLASS_NAMES = [
    "Tomato___Late_blight","Tomato___healthy","Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)","Soybean___healthy",
    "Squash___Powdery_mildew","Potato___healthy",
    "Corn_(maize)___Northern_Leaf_Blight","Tomato___Early_blight",
    "Tomato___Septoria_leaf_spot",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Strawberry___Leaf_scorch","Peach___healthy","Apple___Apple_scab",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Bacterial_spot",
    "Apple___Black_rot","Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew","Peach___Bacterial_spot",
    "Apple___Cedar_apple_rust","Tomato___Target_Spot",
    "Pepper,_bell___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___Late_blight","Tomato___Tomato_mosaic_virus",
    "Strawberry___healthy","Apple___healthy","Grape___Black_rot",
    "Potato___Early_blight","Cherry_(including_sour)___healthy",
    "Corn_(maize)___Common_rust_","Grape___Esca_(Black_Measles)",
    "Raspberry___healthy","Tomato___Leaf_Mold",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Pepper,_bell___Bacterial_spot","Corn_(maize)___healthy"
]

# =========================
# Helper Functions
# =========================
def prettify_label(label):
    return label.replace("___", " → ").replace("_", " ")

def image_to_bytes(image_np):
    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# =========================
# Load Model (cached)
# =========================
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(
        torch.load("model/resnet18_plant_disease.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()

# =========================
# Image Preprocessing
# =========================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# =========================
# Grad-CAM
# =========================
def generate_gradcam(model, input_tensor, target_layer):
    model.eval()
    torch.set_grad_enabled(True)

    activations = None

    def forward_hook(_, __, output):
        nonlocal activations
        activations = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(forward_hook)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    grads = activations.grad
    handle.remove()

    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * activations).sum(dim=1)
    cam = torch.relu(cam).squeeze()
    cam -= cam.min()
    cam /= cam.max()

    return cam.detach().cpu().numpy()

def overlay_gradcam(image, cam, alpha):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)

# =========================
# Upload UI
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload a leaf image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    input_tensor = preprocess_image(image)
    input_tensor.requires_grad_(True)

    # Prediction
    with st.spinner("🧠 Analyzing leaf patterns..."):
        time.sleep(0.8)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    pred_label = CLASS_NAMES[pred_idx]
    pretty_label = prettify_label(pred_label)

    # =========================
    # Prediction Card (FIXED VISIBILITY)
    # =========================
    if "healthy" in pred_label.lower():
        card_bg = "#e6f4ea"
        card_text = "#1e7e34"
        status_text = "🌱 The plant appears healthy."
    else:
        card_bg = "#fdecea"
        card_text = "#842029"
        status_text = "🚨 Signs of plant disease detected."

    st.markdown("### 🧠 Prediction")
    st.markdown(
        f"""
        <div style="
            padding:1.2rem;
            border-radius:12px;
            background-color:{card_bg};
            color:{card_text};
            font-size:18px;
            font-weight:600;
        ">
            {pretty_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(int(confidence*100))

    if confidence < 0.7:
        st.warning("⚠️ Low confidence prediction. Manual verification recommended.")

    st.info(status_text)

    # =========================
    # Probability Distribution
    # =========================
    st.markdown("### 📊 Top Predictions")
    topk = torch.topk(probs, 5)

    fig, ax = plt.subplots()
    labels = [prettify_label(CLASS_NAMES[i]) for i in topk.indices]
    values = topk.values.numpy()

    ax.barh(labels[::-1], values[::-1])
    ax.set_xlim(0,1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)

    # =========================
    # Grad-CAM Section
    # =========================
    st.markdown("---")
    show_xai = st.toggle("🔍 Explain prediction with Grad-CAM")

    if show_xai:
        alpha = st.slider("Heatmap Intensity", 0.1, 0.9, 0.45)

        with st.spinner("Generating explanation..."):
            time.sleep(0.6)
            cam = generate_gradcam(model, input_tensor, model.layer4)

        img_np = np.array(image.resize((224,224)))
        overlay = overlay_gradcam(img_np, cam, alpha)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_np, caption="Original Image")
        with c2:
            st.image(overlay, caption="Grad-CAM Explanation")

        st.info(
            "Grad-CAM highlights regions that influenced the model’s decision. "
            "This is a coarse explanation, not exact lesion segmentation."
        )

        img_bytes = image_to_bytes(overlay)
        filename = f"gradcam_{pred_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        st.download_button(
            "⬇️ Download Grad-CAM Image",
            img_bytes,
            file_name=filename,
            mime="image/png"
        )

    # =========================
    # Model Limitations
    # =========================
    with st.expander("ℹ️ Model Limitations"):
        st.markdown("""
        - Trained on **PlantVillage dataset** (lab-controlled images)
        - Performance may vary on real-world field images
        - Grad-CAM provides **interpretability**, not proof of correctness
        """)

else:
    st.info("👆 Upload a leaf image to begin analysis.")
