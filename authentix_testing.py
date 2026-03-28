import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import os
import zipfile
import tempfile

# --- Configuration ---
# Pointing exactly to the directory containing 'data.pkl' and 'data/' based on your screenshot
MODEL_PATH = r'New folder/authentix_phase1_945.pth/authentix_phase1_945'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Fake vs Real Classifier", layout="centered")


# =====================================================================
# ⚠️ STOP! YOU MUST PASTE YOUR MODEL CLASS DEFINITION HERE ⚠️
# Because you saved a state_dict, PyTorch needs the blueprint of your model.
# Replace the dummy class below with the actual class you used during training.
# =====================================================================
class AuthentixClassifier(nn.Module):
    def __init__(self):
        super(AuthentixClassifier, self).__init__()
        # TODO: Paste your Conv2d, Linear, Dropout layers etc. here
        pass

    def forward(self, x):
        # TODO: Paste your forward pass logic here
        return x


# =====================================================================

@st.cache_resource
def load_classifier():
    model_path = MODEL_PATH
    try:
        # Check if we are dealing with an unzipped PyTorch model directory
        if os.path.isdir(model_path) and 'data.pkl' in os.listdir(model_path):
            st.info("Detected unzipped model directory. Repacking into a valid PyTorch archive...")

            # Create a temporary file to hold the zipped model
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
            tfile.close()
            temp_zip_path = tfile.name

            # Recreate the zip structure PyTorch expects
            with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_STORED) as zf:
                for root, _, files in os.walk(model_path):
                    for file in files:
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, model_path)
                        arcname = os.path.join('archive', rel_path)
                        zf.write(abs_path, arcname)

            model_path = temp_zip_path
            st.success("Model successfully repacked for loading!")

        # 1. Instantiate the "skeleton" of your model
        # Make sure this matches the class name you defined above!
        model = AuthentixClassifier()

        # 2. Load the dictionary of weights (state_dict) from the file
        # We can turn weights_only=True back on since a state_dict is highly secure!
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)

        # 3. Pour the weights into the model skeleton
        model.load_state_dict(state_dict)

        # 4. Now it's a real model, push it to the GPU and set eval mode
        model.to(DEVICE)
        model.eval()
        return model

    except Exception as e:
        st.error(f"Error loading PyTorch model: {e}")
        return None


model = load_classifier()

# --- Preprocessing Pipeline ---
# Adjust the mean, std, and size based on what you used during training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Dashboard UI ---
st.title("🔍 Fake vs Real Image Classifier")
st.write("Upload an image to test the PyTorch `authentix_phase1_945` model.")

tab1, tab2 = st.tabs(["🖼️ Insert Photo", "ℹ️ About Model"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        st.write("Analyzing...")

        if model is not None:
            # Preprocess the image
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_batch)

                # Assuming binary classification with a single output node using Sigmoid:
                prob = torch.sigmoid(output).item()

                # If using CrossEntropy with 2 output nodes, uncomment and adjust:
                # probabilities = F.softmax(output, dim=1)
                # prob = probabilities.item() # Adjust index based on which class is 'Real'

            if prob > 0.5:
                label = "REAL"
                confidence = prob * 100
                st.success(f"### Classification: {label}")
            else:
                label = "FAKE"
                confidence = (1 - prob) * 100
                st.error(f"### Classification: {label}")

            st.write(f"**Confidence Score:** {confidence:.2f}%")
            st.progress(int(confidence))

with tab2:
    st.write("### Model Details")
    st.write("**Model Version:** Phase 1 (945)")
    st.write("**Framework:** PyTorch")
    st.write(f"**Inference Device:** {DEVICE}")