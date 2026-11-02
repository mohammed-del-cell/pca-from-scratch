# PCA from Scratch on Handwritten Digits

I implemented Principal Component Analysis (PCA) with NumPy, checked it against scikit-learn, and tested how many components are needed before a classifier loses accuracy.

**Question:** the digits dataset has 64 features per image. How many can I drop with PCA without hurting a classifier?

## Data

`sklearn.datasets.load_digits`: 1797 images of handwritten digits (8×8 pixels = 64 features, 10 classes). Split: 80% train (1437) / 20% test (360), stratified, `random_state=0`.

## What I built

`src/pca.py` contains a `PCA` class with `fit`, `transform`, `fit_transform` and `inverse_transform`:

1. Center the data (the mean is learned in `fit` and reused in `transform`).
2. Compute the covariance matrix (divided by m − 1).
3. Find its eigenvalues and eigenvectors with `np.linalg.eigh`.
4. Sort from largest to smallest eigenvalue and keep the top k eigenvectors.
5. Project: `Y = (X − mean) @ components`.

**Check against scikit-learn:** on all 1797 digits with k = 10, my eigenvalues, variance ratios, projected data and components match `sklearn.decomposition.PCA` (`np.allclose`; components compared up to sign, because an eigenvector and its negative describe the same direction).

## Results

### How many components keep how much variance?
Each added component adds more variance, but the eigenvalues are sorted from largest to smallest, so the last components add very little information. 21 components keep 90% of the variance, and 29 keep 95%.

21 components keep 90% of the variance, and 29 keep 95%.

![explained variance](images/explained_variance.png)

### What the compression looks like

Rebuilt digits with k = 21 (second row) are still easy to read. They are a little blurrier than the originals (first row).

![rebuilt digits](images/rebuilt_digits_k21.png)

### Accuracy with and without PCA

Logistic regression on the PCA features, 5-fold stratified cross-validation on the training set. PCA is fitted inside each fold on the training part only, so no information from the validation part leaks into it.

| Features | Mean accuracy | Std |
|---|---|---|
| All 64 pixels (no PCA) | 0.962 | 0.012 |
| k = 2 | 0.618 | 0.012 |
| k = 5 | 0.852 | 0.018 |
| k = 10 | 0.925 | 0.012 |
| k = 21 | 0.945 | 0.014 |
| k = 29 | 0.955 | 0.008 |
| k = 40 | 0.963 | 0.010 |
| k = 64 | 0.965 | 0.008 |

![accuracy vs k](images/accuracy_vs_k.png)

I chose **k = 29**, the smallest k whose mean accuracy is within one standard deviation of the no-PCA baseline (k = 21 is clearly below it). It also keeps 95% of the variance.

**Final check on the test set (k = 29):** accuracy 0.958, the same as the model without PCA (0.958).

## Conclusion

On this dataset, 29 PCA components (instead of 64 pixels) give the same accuracy as using all pixels. PCA did **not improve** accuracy. It gave the same accuracy with fewer than half of the features.

## Limitations

- Small and easy dataset (8×8 images). A logistic regression is already fast on 64 features, so the practical gain here is small. PCA matters more with hundreds or thousands of features.
- The test set has 360 images, so one image is about 0.28% accuracy. Differences of one point are within noise.
- The pixels were not standardized, and I only tested one classifier (logistic regression).
- PCA is linear and ignores the labels, so high variance does not always mean useful for the prediction. Here the two agree, but that is not guaranteed on other data.
- The first accuracy table (before cross-validation) used test scores for several values of k, so it was exploratory. k was chosen from the cross-validation results.

## Project structure

```
pca-from-scratch/
├── README.md
├── notebooks/analysis.ipynb
├── src/pca.py
└── images/
```

## How to run

```
pip install numpy scikit-learn matplotlib jupyter
jupyter notebook notebooks/analysis.ipynb
```

Run the notebook from inside the `notebooks` folder, because it imports `src/pca.py` with a relative path.