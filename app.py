"""Agri Vision — Streamlit crop disease classifier."""

from __future__ import annotations

from pathlib import Path
import re

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (APP_DIR / "model" / "best.pt", APP_DIR / "best.pt")


def find_model() -> Path:
    """Support both the recommended model/ folder and a project-root model."""
    for path in MODEL_CANDIDATES:
        if path.is_file():
            return path
    expected = " or ".join(str(path.relative_to(APP_DIR)) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"Model not found. Add best.pt at {expected}.")


@st.cache_resource(show_spinner="Loading disease-detection model…")
def load_model():
    model_path = find_model()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 128)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preprocessing = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, class_names, preprocessing


def readable_label(raw_name: str) -> tuple[str, str, bool]:
    """Convert the original PlantVillage folder name to friendly display text."""
    crop, condition = raw_name.split("___", maxsplit=1)
    crop = crop.replace("_", " ").replace("(including sour)", "(including sour)")
    condition = condition.replace("_", " ").strip()
    condition = re.sub(r"\s+", " ", condition)
    healthy = condition.lower() == "healthy"
    return crop, condition, healthy


def predict(image: Image.Image):
    model, class_names, preprocessing = load_model()
    tensor = preprocessing(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor)[0], dim=0)
    confidence, index = torch.max(probabilities, dim=0)
    return class_names[index.item()], confidence.item()


st.set_page_config(page_title="Agri Vision | Crop Disease Detection", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #f6f8f2; }
      .block-container { max-width: 1040px; padding-top: 2.5rem; padding-bottom: 3rem; }
      .hero { text-align: center; margin-bottom: 2rem; }
      .hero h1 { color: #164a35; font-size: clamp(2rem, 5vw, 3.5rem); margin-bottom: .3rem; }
      .hero p { color: #547062; font-size: 1.08rem; }
      [data-testid="stFileUploader"] { border: 2px dashed #96bda0; border-radius: 18px; background: #fbfdf8; padding: 1rem; }
      [data-testid="stFileUploader"] section { padding: 2rem 1rem; }
      .plus { display:inline-flex; width:54px; height:54px; align-items:center; justify-content:center;
              border-radius:50%; background:#e1f0df; color:#21623f; font-size:2.4rem; line-height:1; margin-bottom:.5rem; }
      .result-card { border-radius:18px; padding:1.5rem 1.75rem; background:#ffffff; border:1px solid #dce8dd; }
      .healthy { border-left: 6px solid #30914c; }
      .disease { border-left: 6px solid #d17b28; }
      .result-card h2 { color:#164a35; margin:.2rem 0 .4rem; }
      .muted { color:#61746a; }
      .stButton > button { width:100%; border-radius:10px; padding:.65rem; font-weight:700;
                             background:#21623f; color:white; border:0; }
      .stButton > button:hover { background:#164a35; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div style="font-size:2.5rem">🌿</div>
      <h1>Agri Vision</h1>
      <p>Crop disease detection from a leaf image</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

left, right = st.columns((1, 1), gap="large")
with left:
    st.markdown('<div class="plus">+</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a clear image of a crop leaf",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a well-lit, close-up image for the best result.",
    )

with right:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
    else:
        st.info("Your uploaded leaf photo will appear here.")

st.divider()
predict_clicked = st.button("Predict disease", disabled=uploaded_file is None, type="primary")

if predict_clicked and uploaded_file:
    with st.spinner("Examining the leaf…"):
        disease_name, confidence = predict(image)
    crop, condition, healthy = readable_label(disease_name)
    card_class = "healthy" if healthy else "disease"
    status = "Healthy crop detected" if healthy else "Disease detected"
    icon = "✅" if healthy else "⚠️"
    st.markdown(
        f"""
        <div class="result-card {card_class}">
          <div class="muted">{icon} {status}</div>
          <h2>{crop} — {condition}</h2>
          <div class="muted">Model confidence: <strong>{confidence:.1%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(confidence * 100), text=f"Confidence: {confidence:.1%}")
    st.caption("Prediction is a decision-support result. For severe or uncertain symptoms, consult a local crop specialist.")

