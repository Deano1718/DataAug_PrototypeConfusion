from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from torchvision import datasets, transforms

import augmentations



def aug_mix(im, preprocess, mixture_depth=-1, all_ops=1, mixture_width=3, aug_severity=3):
    """Perform AugMix augmentations and compute mixture.
    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.
    Returns:
      mixed: Augmented and mixed image.
    """
    image = im.copy()
    with torch.no_grad():
        aug_list = augmentations.augmentations
        if all_ops:
            aug_list = augmentations.augmentations_all

        ws = np.float32(np.random.dirichlet([1] * mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(preprocess(image))
        for i in range(mixture_width):
            image_aug = image.copy()
            depth = mixture_depth if mixture_depth > 0 else np.random.randint(
              1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * preprocess(image_aug)

        mixed = (1 - m) * preprocess(image) + m * mix
        return mixed

class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, js_loss=True, final_process=True,pipelength=1):
      self.dataset = dataset
      self.preprocess = preprocess
      self.js_loss = js_loss
      #preprocess will have normalization if final_process, otherwise, just ToTensor
      self.final_process = final_process
      self.pipelength = pipelength

    def __getitem__(self, i):
        x, y = self.dataset[i]

        if self.final_process:
            if self.js_loss and self.pipelength > 1:
                x0 = transforms.functional.to_pil_image(x[0])
                x1 = transforms.functional.to_pil_image(x[1])
                x2 = transforms.functional.to_pil_image(x[2])

                im_tuple = (self.preprocess(x0), aug_mix(x1, self.preprocess), aug_mix(x2, self.preprocess))
                return im_tuple, y
            elif self.js_loss:
                x0 = transforms.functional.to_pil_image(x)
                im_tuple = (self.preprocess(x0), aug_mix(x0, self.preprocess), aug_mix(x0, self.preprocess))
                return im_tuple, y
            else:
                x0 = transforms.functional.to_pil_image(x)
                return aug_mix(x0, self.preprocess), y
        else:
            if self.js_loss:
                x0 = transforms.functional.to_pil_image(x)
                im_tuple = (self.preprocess(x0), aug_mix(x0, self.preprocess), aug_mix(x0, self.preprocess))
                return im_tuple, y
            else:
                x0 = tranforms.functional.to_pil_image(x)
                return aug_mix(x0, self.preprocess), y

    def __len__(self):
        return len(self.dataset)
