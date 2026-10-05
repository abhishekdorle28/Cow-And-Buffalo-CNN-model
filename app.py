```python
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from datetime import datetime
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Cow vs Buffalo AI Classifier",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #f5f7fb;
}

[data-testid="stSidebar"] {
    background: #111827;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    background: linear-gradient(
        135deg,
        #111827,
        #1f2937,
        #374151
    );

    padding: 35px;
    border-radius: 22px;
    margin-bottom: 25px;
    color: white;

    box-shadow:
        0px 10px 30px rgba(0,0,0,0.12);
}

.hero h1 {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    color: #dbeafe;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;

    box-shadow:
        0px 8px 25px rgba(0,0,0,0.06);

    margin-bottom: 20px;
}

.result-card {
    background: white;
    padding: 30px;
    border-radius: 20px;

    border: 1px solid #e5e7eb;

    box-shadow:
        0px 8px 25px rgba(0,0,0,0.08);
}

.prediction {
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.confidence {
    font-size: 18px;
    color: #475569;
}

.info-box {
    background: #f8fafc;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    font-weight: 700;
    min-height: 48px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = Path("model_quantized.pth")


# =========================================================
# CNN MODEL
# =========================================================

class CowBuffaloCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                123008,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "model_quantized.pth file not found."
        )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = CowBuffaloCNN()


    # -----------------------------
    # Convolution layers
    # -----------------------------

    model.features[0].weight.data.copy_(
        state["features.0.weight"]
    )

    model.features[0].bias.data.copy_(
        state["features.0.bias"]
    )


    model.features[3].weight.data.copy_(
        state["features.3.weight"]
    )

    model.features[3].bias.data.copy_(
        state["features.3.bias"]
    )


    # -----------------------------
    # First Linear Layer
    # -----------------------------

    w1, b1 = state[
        "classifier.1._packed_params._packed_params"
    ]

    model.classifier[1].weight.data.copy_(
        w1.dequantize()
    )

    model.classifier[1].bias.data.copy_(
        b1
    )


    # -----------------------------
    # Second Linear Layer
    # -----------------------------

    w2, b2 = state[
        "classifier.4._packed_params._packed_params"
    ]

    model.classifier[4].weight.data.copy_(
        w2.dequantize()
    )

    model.classifier[4].bias.data.copy_(
        b2
    )


    model.eval()

    return model


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.ToTensor()

])


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(image, model):

    image = image.convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)


    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


    predicted_index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    confidence = float(
        probabilities[
            predicted_index
        ].item()
    )


    return (
        predicted_index,
        confidence,
        probabilities.tolist()
    )


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:

    st.session_state.history = []


if "result" not in st.session_state:

    st.session_state.result = None


# =====================
```
