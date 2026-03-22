import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        self.transform = self._build_transforms()

    def _build_transforms(self):
        if self.augment:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '_segmentation.png')

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)

        return image, mask