# TrustSeg: Uncertainty-Aware Skin Lesion Segmentation

This project builds a deep learning model for skin lesion segmentation and explores how uncertainty can help identify unreliable predictions. The model predicts lesion regions from dermoscopic images and also shows where it is less confident.

## Overview

Skin lesion segmentation is important for early detection of conditions like melanoma. Most models only output predictions, but in real settings it is just as important to know when a model might be wrong.

In this project, I built a segmentation system that:

* predicts lesion masks using a U-Net style model
* estimates uncertainty using Monte Carlo Dropout
* visualizes both predictions and uncertainty

I also explored a key question:
**can uncertainty actually indicate when the model is making mistakes?**

## Approach

The model is implemented in PyTorch and trained on dermoscopic images.

Key components:

* U-Net style convolutional network for segmentation
* Monte Carlo Dropout for uncertainty estimation
* Dice loss for training
* simple data augmentation (flips, rotations, color changes)

During inference, the model runs multiple times on the same image with dropout enabled. The variation across predictions is used to generate an uncertainty map.

## What the model outputs

For each image:

* predicted segmentation mask
* uncertainty map (higher = less confident)

This helps highlight regions where the model may need human review.

## Analysis

To understand whether uncertainty is meaningful, I compared:

* segmentation performance (Dice score)
* average uncertainty per image

This helps evaluate whether higher uncertainty corresponds to lower model accuracy, which is important for building more trustworthy systems.

## Project Structure

SkinSegmentationProject/
├── data/
│   ├── images/
│   ├── masks/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── uncertainty.py
├── app.py
├── requirements.txt
├── README.md

## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Prepare your dataset:

* images → data/images/
* masks → data/masks/

Mask naming should match:
image_001.jpg → image_001_segmentation.png

## Training

python3 -m src.train

This will train the model and save weights to:
model/best_model.pth

## Run the App

streamlit run app.py

Then open:
https://skin-lesion-app-likhitha-battu.streamlit.app/

You can upload an image and view:

* original image
* predicted mask
* uncertainty map

## Model + Uncertainty

The model uses Monte Carlo Dropout to estimate uncertainty.

Instead of making one prediction, it runs multiple forward passes with dropout enabled. The final output includes:

* mean prediction (segmentation)
* variance (uncertainty)

Higher variance means the model is less confident in that region.

## Evaluation

Performance is measured using Dice score, which compares overlap between prediction and ground truth.

## Notes

* The model expects binary masks (0 = background, 1 = lesion)
* Images are resized during preprocessing
* GPU is used if available, otherwise it falls back to CPU

## Future Work

* improve model performance
* explore better uncertainty methods
* add more evaluation metrics
* extend to multi-class segmentation

This project focuses on building a simple but complete pipeline: data → model → uncertainty → visualization, with an emphasis on understanding model reliability.
