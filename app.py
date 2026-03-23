import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.model import UNetMCDropout
from src.uncertainty import mc_predict
from src.active_learning import rank_by_uncertainty

st.set_page_config(page_title="TrustSeg", layout="wide")

st.markdown("""
<style>
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(200, 100, 150, 0.4); }
    50% { box-shadow: 0 0 20px 6px rgba(200, 100, 150, 0.2); }
}

.animated-title {
    background: linear-gradient(270deg, #ff6b9d, #c44dff, #6b9dff, #ff6b9d);
    background-size: 400% 400%;
    animation: gradientShift 6s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: #888;
    font-size: 1.1rem;
    animation: fadeInUp 1s ease forwards;
    margin-bottom: 2rem;
}

.info-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(200, 100, 150, 0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    animation: fadeInUp 0.8s ease forwards;
}

.info-card h3 {
    color: #ff6b9d;
    margin-bottom: 0.8rem;
    font-size: 1.1rem;
}

.info-card ul {
    color: #ccc;
    line-height: 2;
    padding-left: 1.2rem;
}

.result-container {
    animation: fadeInUp 0.6s ease forwards;
}

.uncertainty-badge-high {
    background: linear-gradient(135deg, #ff4444, #cc0000);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
    animation: pulse 2s infinite;
}

.uncertainty-badge-medium {
    background: linear-gradient(135deg, #ffaa00, #cc7700);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}

.uncertainty-badge-low {
    background: linear-gradient(135deg, #00cc66, #009944);
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-block;
}

div.stButton > button {
    background: radial-gradient(150% 180% at 11% 140%, #000 37%, #08012c 61%, #4e1e40 78%, #70464e 89%, #88394c 100%);
    color: white;
    border: 1px solid rgba(220, 100, 150, 0.3);
    border-radius: 11px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.5s ease;
    min-width: 150px;
}

div.stButton > button:hover {
    background: radial-gradient(120% 103% at 0% 91%, #c96287 0%, #c66c64 8%, #cc7d23 21%, #37140a 71%, #000 85%);
    border-color: rgba(220, 150, 180, 0.6);
    transform: scale(1.02);
}

div.stButton > button:active {
    transform: scale(0.98);
}

.stTabs [data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    color: #ff6b9d !important;
    border-bottom: 2px solid #ff6b9d !important;
}

section[data-testid="stFileUploadDropzone"] {
    border: 2px dashed rgba(200, 100, 150, 0.4) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease;
}

section[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(200, 100, 150, 0.8) !important;
    background: rgba(200, 100, 150, 0.05) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="animated-title">TrustSeg</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Uncertainty-Aware Skin Lesion Segmentation using Monte Carlo Dropout</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <h3>What this tool does</h3>
    <ul>
        <li>Segments skin lesions using a lightweight U-Net deep learning model</li>
        <li>Estimates prediction uncertainty using Monte Carlo Dropout</li>
        <li>Highlights unreliable predictions for clinician review</li>
        <li>Ranks unlabeled images by uncertainty for efficient annotation</li>
    </ul>
</div>
<div class="info-card">
    <h3>How to interpret results</h3>
    <ul>
        <li>Low uncertainty — model is confident, prediction is reliable</li>
        <li>Medium uncertainty — moderate confidence, review recommended</li>
        <li>High uncertainty — model is unsure, manual review required</li>
    </ul>
</div>
""", unsafe_allow_html=True)

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

tab1, tab2 = st.tabs(["Segmentation Demo", "Active Learning"])

# ---------------------- TAB 1 ----------------------
with tab1:
    uploaded_file = st.file_uploader("Choose a skin lesion image", type=["jpg", "jpeg", "png"])
    use_sample = st.button("Try a sample image")

    if use_sample:
        image = Image.open("sample.png")
    elif uploaded_file:
        image = Image.open(uploaded_file)
    else:
        image = None

    if image is not None:
        try:
            model, device = load_model()

            with st.spinner("Running 20 stochastic forward passes..."):
                tensor = preprocess(image)
                mean_pred, uncertainty = mc_predict(model, tensor, n_passes=20, device=device)

            st.markdown('<div class="result-container">', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.subheader("Original Image")
                image_display = image.convert("RGB").resize((256, 256))
                st.image(image_display, use_container_width=True)

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

            with col4:
                st.subheader("Overlay")
                image_resized = np.array(image.convert("RGB").resize((256, 256)))
                mask_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
                mask_binary = (mean_pred > 0.5)
                mask_rgb[mask_binary] = [255, 0, 0]
                overlay = (0.6 * image_resized + 0.4 * mask_rgb).astype(np.uint8)
                st.image(overlay, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

            avg_uncertainty = uncertainty.mean()

            st.divider()

            if avg_uncertainty < 0.01:
                badge = '<span class="uncertainty-badge-low">High Confidence</span>'
            elif avg_uncertainty < 0.05:
                badge = '<span class="uncertainty-badge-medium">Medium Confidence</span>'
            else:
                badge = '<span class="uncertainty-badge-high">Low Confidence</span>'

            st.markdown(f"**Model Confidence:** {badge} &nbsp;&nbsp; **Average Uncertainty:** `{avg_uncertainty:.6f}`", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if avg_uncertainty >= 0.03:
                st.warning("Low confidence prediction — recommend manual review by a clinician.")

            if avg_uncertainty < 0.01:
                st.success("Model is very confident in this prediction.")
            elif avg_uncertainty < 0.03:
                st.info("Moderate confidence. Review recommended.")
            else:
                st.error("Low confidence. Prediction may be unreliable.")

            st.divider()
            st.subheader("Pixel-Level Uncertainty Distribution")
            st.markdown("This histogram shows how uncertainty is distributed across every pixel. A spike on the left means most pixels are confident. A long right tail means many pixels are uncertain.")

            fig2, ax2 = plt.subplots(figsize=(8, 3))
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_facecolor('#0e1117')
            uncertainty_flat = uncertainty.flatten()
            ax2.hist(uncertainty_flat, bins=80, color='#ff6b9d', alpha=0.8, edgecolor='none')
            ax2.axvline(avg_uncertainty, color='white', linestyle='--', linewidth=1.5, label=f'Mean: {avg_uncertainty:.4f}')
            ax2.axvline(0.03, color='orange', linestyle='--', linewidth=1.5, label='Review threshold (0.03)')
            ax2.set_xlabel("Uncertainty (Variance)", color='white')
            ax2.set_ylabel("Pixel Count", color='white')
            ax2.set_title("Distribution of Pixel-Level Uncertainty", color='white')
            ax2.tick_params(colors='white')
            ax2.spines['bottom'].set_color('#444')
            ax2.spines['left'].set_color('#444')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.legend(facecolor='#1a1a2e', labelcolor='white')
            st.pyplot(fig2)
            plt.close()

            st.divider()
            st.subheader("Model Performance Analysis")
            st.markdown("Uncertainty vs Dice score across **800 training images**. Higher uncertainty predicts lower segmentation accuracy.")

            if os.path.exists("uncertainty_vs_dice.png"):
                st.image("uncertainty_vs_dice.png", use_container_width=True, caption="Negative correlation between MC Dropout uncertainty and Dice coefficient (n=800)")
            else:
                st.info("Run src/evaluate.py to generate this plot.")

        except FileNotFoundError:
            st.warning("No trained model found. Please train the model first by running src/train.py")

# ---------------------- TAB 2 ----------------------
with tab2:
    st.subheader("Uncertainty-Guided Active Learning")
    st.markdown("""
    Upload multiple unlabeled images. The model ranks them by uncertainty.
    Label the most uncertain ones first to improve performance efficiently.

    This implements **maximum entropy sampling**, a core strategy in active learning research.
    """)

    uploaded_files = st.file_uploader(
        "Upload unlabeled images to rank by uncertainty",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        try:
            model, device = load_model()

            with st.spinner(f"Ranking {len(uploaded_files)} images by uncertainty..."):
                images = [(f.name, Image.open(f)) for f in uploaded_files]
                ranked = rank_by_uncertainty(model, images, device, n_passes=10)

            st.success(f"Ranked {len(ranked)} images. Label the top ones first!")
            st.divider()

            for i, result in enumerate(ranked):
                u = result['uncertainty']
                if u > 0.03:
                    border_color = "rgba(255, 68, 68, 0.7)"
                    bg_color = "rgba(255, 68, 68, 0.08)"
                    priority = "High priority"
                    dot_color = "#ff4444"
                elif u > 0.01:
                    border_color = "rgba(255, 170, 0, 0.7)"
                    bg_color = "rgba(255, 170, 0, 0.08)"
                    priority = "Medium priority"
                    dot_color = "#ffaa00"
                else:
                    border_color = "rgba(0, 204, 102, 0.7)"
                    bg_color = "rgba(0, 204, 102, 0.08)"
                    priority = "Low priority"
                    dot_color = "#00cc66"

                st.markdown(f"""
                <div style="
                    border: 1px solid {border_color};
                    background: {bg_color};
                    border-radius: 12px;
                    padding: 0.8rem 1.2rem;
                    margin-bottom: 10px;
                    display: flex;
                    align-items: center;
                    gap: 12px;
                ">
                    <div style="width:12px; height:12px; border-radius:50%; background:{dot_color}; flex-shrink:0;"></div>
                    <span style="font-weight:600;">#{i+1} — {result['name']}</span>
                    <span style="color:#aaa; font-size:0.9rem;">Uncertainty: {result['uncertainty']:.6f}</span>
                    <span style="margin-left:auto; font-size:0.85rem; color:{dot_color}; font-weight:600;">{priority}</span>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.caption("Original image")
                    st.image(result['image'], use_container_width=True)

                with c2:
                    st.caption("Predicted mask")
                    mask_display = (result['mean_pred'] > 0.5).astype(np.uint8) * 255
                    st.image(mask_display, use_container_width=True, clamp=True)

                with c3:
                    st.caption("Uncertainty map")
                    fig3, ax3 = plt.subplots()
                    fig3.patch.set_facecolor('#0e1117')
                    ax3.set_facecolor('#0e1117')
                    ax3.imshow(result['uncertainty_map'], cmap='RdYlGn_r')
                    ax3.axis('off')
                    st.pyplot(fig3)
                    plt.close()

                st.divider()

            st.subheader("Uncertainty Rankings Summary")
            fig4, ax4 = plt.subplots(figsize=(8, 3))
            fig4.patch.set_facecolor('#0e1117')
            ax4.set_facecolor('#0e1117')
            names = [r['name'][:15] for r in ranked]
            uncertainties = [r['uncertainty'] for r in ranked]
            colors = ['#ff4444' if u > 0.03 else '#ffaa00' if u > 0.01 else '#00cc66' for u in uncertainties]
            ax4.barh(names, uncertainties, color=colors)
            ax4.axvline(0.03, color='white', linestyle='--', linewidth=1, label='Review threshold')
            ax4.set_xlabel("Average Uncertainty", color='white')
            ax4.set_title("Images Ranked by Uncertainty", color='white')
            ax4.tick_params(colors='white')
            ax4.spines['bottom'].set_color('#444')
            ax4.spines['left'].set_color('#444')
            ax4.spines['top'].set_visible(False)
            ax4.spines['right'].set_visible(False)
            ax4.legend(facecolor='#1a1a2e', labelcolor='white')
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

        except FileNotFoundError:
            st.warning("No trained model found. Please train the model first.")