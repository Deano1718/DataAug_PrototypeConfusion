# DataAug_PrototypeConfusion

Provides implementation of image data augmentation strategy that replaces windows of pixels with high confidence regions of opposing classes.  Additional provides manual implementation of Augmix and PRIME, which can be used in conjunction.

Augmix Reference: 
```
@article{hendrycks2020augmix,
  title={{AugMix}: A Simple Data Processing Method to Improve Robustness and Uncertainty},
  author={Hendrycks, Dan and Mu, Norman and Cubuk, Ekin D. and Zoph, Barret and Gilmer, Justin and Lakshminarayanan, Balaji},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

PRIME Reference:
```
@inproceedings{PRIME2022,
    title = {PRIME: A Few Primitives Can Boost Robustness to Common Corruptions}, 
    author = {Apostolos Modas and Rahul Rade and Guillermo {Ortiz-Jim\'enez} and Seyed-Mohsen {Moosavi-Dezfooli} and Pascal Frossard},
    year = {2022},
    booktitle = {European Conference on Computer Vision (ECCV)}
}
```
