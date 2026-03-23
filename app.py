import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.model import UNetMCDropout
from src.uncertainty import mc_predict

st.set_page_config(page_title="TrustSeg", layout="wide")

st.title("TrustSeg: Uncertainty-Aware Skin Lesion Segmentation")
st.markdown("Upload a dermoscopic image to see the predicted segmentation mask and uncertainty map.")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetMCDropout(dropout_p=0.3).to(device)
    model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
    return model, device

def preprocess(image):
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    image_np = np.array(image.convert("RGB"))
    return transform(image=image_np)['image']

uploaded_file = st.file_uploader("Choose a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    try:
        model, device = load_model()
        
        with st.spinner("Running 20 stochastic forward passes..."):
            tensor = preprocess(image)
            mean_pred, uncertainty = mc_predict(model, tensor, n_passes=20, device=device)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Predicted Mask")
            mask_display = (mean_pred > 0.5).astype(np.uint8) * 255
            st.image(mask_display, use_container_width=True, clamp=True)

        with col3:
            st.subheader("Uncertainty Map")
            fig, ax = plt.subplots()
            ax.imshow(uncertainty, cmap='RdYlGn_r')
            ax.axis('off')
            plt.colorbar(ax.images[0], ax=ax, fraction=0.046)
            st.pyplot(fig)
            plt.close()

        avg_uncertainty = uncertainty.mean()
        dice_approx = 1 - avg_uncertainty * 100

        st.divider()
        col4, col5 = st.columns(2)
        with col4:
            st.metric("Average Uncertainty", f"{avg_uncertainty:.6f}")
        with col5:
            confidence = "High" if avg_uncertainty < 0.01 else "Medium" if avg_uncertainty < 0.05 else "Low"
            st.metric("Model Confidence", confidence)

    except FileNotFoundError:
        st.warning("No trained model found. Please train the model first by running src/train.py")