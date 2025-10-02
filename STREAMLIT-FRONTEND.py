%%writefile app.py
import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
from PIL import Image
import numpy as np
import io
import torch
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import base64
import gdown

def display_centered_images(original_path, masked_path):
    # Outer 3-column layout to center the images block
    outer_cols = st.columns([1, 2, 1])
    with outer_cols[1]:
        # Inner 2-column layout to place images close together
        inner_cols = st.columns(2, gap="small")
        with inner_cols[0]:
            st.image(original_path, caption="Original Image", use_container_width=True)
        with inner_cols[1]:
            st.image(masked_path, caption="Masked Image", use_container_width=True)

# ------------------ Streamlit Page Config ------------------
st.set_page_config(
    page_title="🎨 AI Object Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Sidebar ------------------
st.sidebar.header("Customize Dashboard")
bg_color = st.sidebar.color_picker("Dashboard Background", "#E6E6FA")
sidebar_start = st.sidebar.color_picker("Sidebar Gradient Start", "#FFB6C1")
sidebar_end = st.sidebar.color_picker("Sidebar Gradient End", "#87CEFA")

st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image (JPG, JPEG, PNG, etc.)",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
    accept_multiple_files=False
)

st.sidebar.markdown(
    """
    <div style="font-size:20px; font-weight:bold; font-family:'Garamond', 'Times New Roman', serif;color:#000000; line-height:1.7;">
        <strong>Instructions:</strong><br>
        - Upload one image at a time.<br>
        - See original and masked images side by side.<br>
        - Use overlay slider to blend images.<br>
        - Download masked image.<br>
        - Download overlay image.
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------ Custom CSS ------------------
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: {bg_color};
}}
h1 {{
    text-align: center;
    color: white;
    font-family: 'Arial', sans-serif;
}}
.css-1d391kg {{
    background: linear-gradient(to bottom, {sidebar_start}, {sidebar_end});
    color: white;
    font-weight: bold;
}}
h2, h3 {{
    color: #4B0082;
}}
.stButton button {{
    background-color: #FF69B4;
    color: white;
    border-radius: 10px;
    height: 40px;
    width: 220px;
    font-size: 16px;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# ------------------ Header ------------------
st.markdown(f"""
<div style='text-align: center; padding: 20px;
            background: linear-gradient(to right, {sidebar_start}, {sidebar_end});
            color: white; border-radius:10px;' >
    <h1>🤖 AI Object Extractor Dashboard</h1>
    <p style='text-align: center; font-size:18px; color:white;' >
    Upload images, extract objects, and download results!</p>
    <hr style='border:2px solid white'>
</div>
""", unsafe_allow_html=True)


# ------------------ Model Setup ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(pretrained=False, num_classes=2)

# Google Drive model info
GDRIVE_FILE_ID = "1J5WRWblZvUeReVRxy92wrH1zyh0navVH"  # Your public file ID
MODEL_PATH = "best_deeplabv3_finetuned.pth"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id=1J5WRWblZvUeReVRxy92wrH1zyh0navVH"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model weights
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
except:
    st.error("Failed to load the model. Make sure the Google Drive file is a valid .pth file.")
    st.stop()

# ------------------ Helper Functions ------------------
resize_input_size = (256, 256)

def preprocess_image(img):
    transform = T.Compose([
        T.Resize(resize_input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor.to(DEVICE), img

def clean_mask(mask):
    mask_uint8 = (mask*255).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    clean = cv2.GaussianBlur(clean, (7,7), 0)
    return clean

def apply_mask_to_image(original_img, mask):
    mask_resized = cv2.resize(mask, (original_img.width, original_img.height), interpolation=cv2.INTER_LINEAR)
    img_array = np.array(original_img)
    out = np.zeros_like(img_array)
    out[mask_resized>127] = img_array[mask_resized>127]
    return Image.fromarray(out)

# ------------------ Demo Preview Section ------------------
# ------------------ Demo Preview Section ------------------
st.markdown("## 🔹 Sample Preview")
st.markdown(
    '<div style="text-align: center; font-size: 18px;font-weight: bold; color: #964B00;margin-top: 10px;">'
    ' Pixel-perfect AI masking in a click!✨'
    '</div>',
    unsafe_allow_html=True
)

demo_path = "/content/drive/MyDrive/AI-Vision/images/puppy.jpg"
masked_path = "/content/drive/MyDrive/AI-Vision/images/masked_puppy.jpg"

display_centered_images(demo_path, masked_path)




# ------------------ Main App ------------------
if uploaded_file:
    original_image = Image.open(uploaded_file).convert("RGB")
    img_tensor, _ = preprocess_image(original_image)

    with torch.no_grad():
        output = model(img_tensor)['out']
        mask_raw = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    mask_cleaned = clean_mask(mask_raw)
    masked_image = apply_mask_to_image(original_image, mask_cleaned)

    # Display Original and Masked
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_image)
    with col2:
        st.subheader("Masked Image")
        st.image(masked_image)
        buf = io.BytesIO()
        masked_image.save(buf, format="PNG")
        st.download_button("Download Masked Image", buf.getvalue(), file_name=f"masked_{uploaded_file.name}")

    # Overlay
    st.subheader("Overlay Image")
    opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5)
    overlay = Image.blend(original_image, masked_image, alpha=opacity)
    st.image(overlay, caption=f'Overlay ({opacity*100:.0f}%)')
    buf_overlay = io.BytesIO()
    overlay.save(buf_overlay, format="PNG")
    st.download_button("Download Overlay Image", buf_overlay.getvalue(), file_name=f"overlay_{uploaded_file.name}")
else:
    st.info("Please upload an image to start processing.")
