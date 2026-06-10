# H&E Tissue Classification with CNNs

**Author:** Parul Kulshreshtha

This project addresses *AI in Pathology Image Analysis*. Eight haematoxylin & eosin (H&E) stained images are tiled into 100×100 patches and used to train two binary convolutional neural networks, which are then cross-tested against each other's data.

## Overview

The notebook (`Submission_ready_LAQ3.html`) runs an end-to-end image-analysis pipeline:

1. Load 8 H&E slides organised into four groups (A, B, C, D).
2. Cut each slide into 100×100 pixel tiles and discard tiles that are mostly background (tissue filter at a 20% threshold).
3. Split each model's tiles into a 90:10 train/validation split (stratified).
4. Train two CNNs with data augmentation:
   - **Model 1** — classifies group **A vs B**
   - **Model 2** — classifies group **C vs D**
5. Evaluate each model on its own validation set, then cross-test each model on the *other* model's data.

## Data

Eight `.jpg` files, grouped as follows:

| Group | Files  | Used by  |
|-------|--------|----------|
| A     | A1, A2 | Model 1  |
| B     | B1, B2 | Model 1  |
| C     | C1, C2 | Model 2  |
| D     | D1, D2 | Model 2  |

The images are small pre-cropped regions (~900×1300 px) rather than whole-slide images. D2 is smaller (688×1164 px), producing fewer tiles and a mild class imbalance in group D.

In the notebook the data is loaded from a local path (`base_path` in code block 2). Update this path to point to your own copy of the slides before running.

## Pipeline structure

The notebook is organised into numbered code blocks:

- **Block 1** — Imports and helper functions (`tissue_percent`, tiling, versioned output paths). All outputs are namespaced under a `PIPELINE_VERSION` tag (`outputs_v2/`) so a re-run won't overwrite previous results.
- **Block 2** — Load all 8 images (BGR→RGB conversion).
- **Block 3** — Cut images into 100×100 tiles with tissue filtering.
- **Block 4** — Save tiles to disk.
- **Block 5** — Build DataFrames of tile paths and labels (label parsed from filename prefix).
- **Block 6** — Stratified 90:10 train/validation split.
- **Block 7** — Keras `ImageDataGenerator` data generators (augmentation on train only).
- **Block 8** — CNN model definitions (3 Conv2D blocks + classifier head, batch norm, dropout).
- **Block 9 / 10** — Train Model 1 and Model 2, with early stopping and best-weight checkpointing.
- **Block 11** — Evaluate both models (answers Q1 and Q2).
- **Block 12** — Cross-dataset testing (answers Q4).
- Plus interspersed cells for stain fingerprinting, augmentation visualisation, training-history plots, and detailed cross-test analysis.

## Model architecture

Both models use the same skeleton: three convolutional blocks (32 → 64 → 128 filters, 3×3 kernels, `same` padding, ReLU, He initialisation), each followed by 2×2 max pooling and batch normalisation, then a flatten + dropout + dense classifier head. Training uses the Adam optimiser, batch size 16, up to 30 epochs with early stopping (patience 7, restore best weights). Seeds are fixed (`42`) for reproducibility.

Augmentation (training set only): rescale to [0,1], horizontal/vertical flips, 90° rotation range, 0.2 height shift, reflect fill.

## Requirements

- Python 3
- `tensorflow` / `keras`
- `opencv-python` (`cv2`)
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib`
- `Pillow`

## How to run

The submission is provided as a rendered HTML export of the notebook. To re-run it:

1. Recover the `.ipynb` (or recreate the cells) and open it in Jupyter.
2. Set `base_path` in code block 2 to your local data directory.
3. Run the cells top to bottom. Outputs (tiles, saved models such as `model_AB_best.keras` / `model_CD_best.keras`) are written under `outputs_v2/`.

## Results summary

| Model | Dataset | Training accuracy | Validation accuracy |
|-------|---------|-------------------|---------------------|
| Model 1 | A vs B | ~96.8% | ~97.6% |
| Model 2 | C vs D | ~100% | ~100% |

Model 2 reached perfect validation accuracy and converged very quickly, while Model 1 converged gradually. The notebook argues Model 2's apparent superiority reflects greater between-group stain separability (groups C and D differ markedly in colour, whereas A and B are nearly identical) rather than genuinely better feature learning — i.e. the network likely latched onto a low-level colour cue.

**Cross-testing (Q4):** when each model was applied to the other dataset, accuracy collapsed to roughly chance level (~46% and ~52%). The notebook attributes this to label-space mismatch, feature-space drift between domains, reliance on dataset-specific shortcuts, limited training data, and the absence of any domain-adaptation strategy (Q5).

## Notes & caveats

- Datasets are small (~400 tiles for Model 1, ~366 for Model 2), so validation accuracy is noisy — a single misclassification shifts it by ~2.5 percentage points.
- The reported accuracies are sensitive to stain/colour differences between groups, so high within-domain accuracy should not be read as robust morphological learning.

## References

The notebook does not contain an explicit reference list; the works below underpin the methods and concepts it uses (CNN classification of H&E images, augmentation, stain variation, shortcut learning) and the libraries it relies on. Check that they fit your required citation style and reading list before submitting.

**Methods and concepts**

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Komura, D., & Ishikawa, S. (2018). Machine learning methods for histopathological image analysis. *Computational and Structural Biotechnology Journal*.
- Tellez, D., et al. (2019). Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. *Medical Image Analysis*.
- Macenko, M., et al. (2009). A method for normalizing histology slides for quantitative analysis. *IEEE International Symposium on Biomedical Imaging (ISBI)*.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *International Conference on Machine Learning (ICML)*.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*.

**Software and libraries**

- Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. *USENIX OSDI*.
- Chollet, F., et al. (2015). *Keras*. https://keras.io
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*.
- Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*.
- Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
- McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference (SciPy)*.
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*.
