import torch
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import ISICDataset
from src.model import UNetMCDropout
from src.uncertainty import mc_predict, dice_score

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ISICDataset(
        image_dir="data/images",
        mask_dir="data/masks",
        img_size=256,
        augment=False
    )

    model = UNetMCDropout(dropout_p=0.3).to(device)
    model.load_state_dict(torch.load("model/best_model.pth", map_location=device))

    dice_scores = []
    uncertainties = []

    for i in range(len(dataset)):
        image, mask = dataset[i]
        mask_np = mask.squeeze().numpy()

        mean_pred, uncertainty = mc_predict(model, image, n_passes=20, device=device)

        dice = dice_score(mean_pred, mask_np)
        avg_uncertainty = uncertainty.mean()

        dice_scores.append(dice)
        uncertainties.append(avg_uncertainty)

        if i % 20 == 0:
            print(f"[{i}/{len(dataset)}] Dice: {dice:.4f} | Uncertainty: {avg_uncertainty:.6f}")

    dice_scores = np.array(dice_scores)
    uncertainties = np.array(uncertainties)

    plt.figure(figsize=(8, 5))
    plt.scatter(uncertainties, dice_scores, alpha=0.6, color='steelblue')
    plt.xlabel("Average Uncertainty (Variance)")
    plt.ylabel("Dice Score")
    plt.title("Uncertainty vs Segmentation Performance")
    plt.savefig("uncertainty_vs_dice.png", dpi=150)
    plt.close()

    print(f"\nMean Dice: {dice_scores.mean():.4f}")
    print(f"Mean Uncertainty: {uncertainties.mean():.6f}")
    print("Saved plot to uncertainty_vs_dice.png")

if __name__ == "__main__":
    evaluate()