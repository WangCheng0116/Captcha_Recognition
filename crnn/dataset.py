import albumentations
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# class ClassificationDataset:
#     def __init__(self, image_paths, targets, resize=None):
#         self.image_paths = image_paths
#         self.targets = targets
#         self.resize = resize
#         self.augmentations = albumentations.Compose(
#             [albumentations.Normalize(always_apply=True)])

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, item):
#         image = Image.open(self.image_paths[item])
#         image = image.convert("RGB")
#         targets = self.targets[item]

#         if self.resize is not None:
#             image = image.resize(
#                 (self.resize[1], self.resize[0]), resample=Image.BILINEAR
#             )

#         image = np.array(image).astype(np.float32)
#         image = np.expand_dims(image, axis=0)

#         augmented = self.augmentations(image=image)
#         image = augmented["image"]

#         return {
#             "images": torch.tensor(image, dtype=torch.float),
#             "targets": torch.tensor(targets, dtype=torch.long),
#         }


class ClassificationDataset:
    def __init__(self, image_paths, targets, resize=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = albumentations.Compose(
            [albumentations.Normalize(always_apply=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = image.convert("L")  # Convert image to grayscale
        targets = self.targets[item]

        # Convert image to numpy array for pixel manipulation
        image = np.array(image).astype(np.float32)

        # Change all black pixels (0) to white (255)
        image[image == 0] = 255
        image[image < 255] = 0

        if self.resize is not None:
            # Convert back to PIL Image for resizing
            image = Image.fromarray(image.astype(np.uint8))
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )
            image = np.array(image).astype(np.float32)

        image = np.expand_dims(image, axis=0)

        augmented = self.augmentations(image=image)
        image = augmented["image"]

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
