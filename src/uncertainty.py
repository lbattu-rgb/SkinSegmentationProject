import torch
import numpy as np
from src.model import UNetMCDropout

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train()

def mc_predict(model, image_tensor, n_passes=20, device='cpu'):
    model.eval()
    enable_dropout(model)
    
    image_tensor = image_tensor.unsqueeze(0).to(device)
    predictions = []

    with torch.no_grad():
        for _ in range(n_passes):
            output = model(image_tensor)
            prob = torch.sigmoid(output)
            predictions.append(prob.cpu().numpy())

    predictions = np.array(predictions)

    mean_pred = predictions.mean(axis=0).squeeze()
    uncertainty = predictions.var(axis=0).squeeze()

    return mean_pred, uncertainty


def dice_score(pred, mask, threshold=0.5):
    pred_binary = (pred > threshold).astype(np.float32)
    intersection = (pred_binary * mask).sum()
    return (2 * intersection + 1) / (pred_binary.sum() + mask.sum() + 1)