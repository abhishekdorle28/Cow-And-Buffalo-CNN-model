import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Cow vs Buffalo Classifier",
    page_icon="🐄",
    layout="centered"
)

st.title("🐄 Cow vs Buffalo AI Classifier")
st.write("Upload an animal image to predict whether it is a Cow or Buffalo.")

# Model path
MODEL_PATH = Path("model_quantized.pth")

# CNN model
class CowBuffaloCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(123008, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model_quantized.pth not found in the project folder."
        )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = CowBuffaloCNN()

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

    w1, b1 = state[
        "classifier.1._packed_params._packed_params"
    ]
    model.classifier[1].weight.data.copy_(w1.dequantize())
    model.classifier[1].bias.data.copy_(b1)

    w2, b2 = state[
        "classifier.4._packed_params._packed_params"
    ]
    model.classifier[4].weight.data.copy_(w2.dequantize())
    model.classifier[4].bias.data.copy_(b2)

    model.eval()
    return model


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Upload image
uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Animal", type="primary"):
        try:
            with st.spinner("Analyzing image..."):
                model = load_model()

                image_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]

                predicted_index = int(torch.argmax(probabilities).item())
                confidence = float(probabilities[predicted_index].item())

                # Assumes class 0 = Cow and class 1 = Buffalo
                labels = ["Cow", "Buffalo"]
                prediction = labels[predicted_index]

                st.session_state.history.append({
                    "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    "Prediction": prediction,
                    "Confidence": f"{confidence * 100:.2f}%"
                })

                st.success("Prediction completed!")

                st.subheader(f"Result: {prediction}")
                st.metric("Confidence", f"{confidence * 100:.2f}%")

                st.write("Class probabilities:")
                st.write(f"Cow: {probabilities[0].item() * 100:.2f}%")
                st.write(f"Buffalo: {probabilities[1].item() * 100:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")

# History
st.divider()
st.subheader("📋 Prediction History")

if st.session_state.history:
    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No predictions yet. Upload an image to get started.")
