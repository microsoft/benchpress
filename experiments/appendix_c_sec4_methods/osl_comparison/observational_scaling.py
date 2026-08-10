"""Observational scaling laws (Ruan et al., 2024), reimplemented for comparison.

Reviewer nxZ2 asks whether the paper's finding is already covered by
Observational Scaling Laws (OSL), and how the prediction recipe compares to it.
Answering that needs OSL as a runnable predictor, so this module reimplements
its fitting procedure from the paper and from the reference implementation at
github.com/ryoungj/ObsScaling (`utils/data.py`).

The recipe has three stages, and each one is reproduced here:

1. `pca_impute` fills missing entries of the base benchmark block. Following the
   reference implementation, columns are standardized, missing cells are seeded
   with the column mean, a rank-`n_components` PCA reconstruction is applied to
   the missing cells until the reconstruction stops moving or 1000 iterations
   are reached, and the result is clipped to each column's metric range. The
   scaler, the mean seed, and the PCA are all fitted on the training rows only
   and then applied to the test rows, which is how OSL prevents train-test
   leakage.
2. `fit_capability_pca` extracts the top `n_components` principal components of
   the imputed block. OSL mean-centers without rescaling here, because the base
   metrics are already on a common [0, 1] accuracy scale, so this stage uses a
   separate PCA with `standardize=False`.
3. `SigmoidCapabilityRegression` fits the target metric as
   `E = (1 - b) * sigmoid(beta' S + alpha) + b`. This is the `sigmoid-parametric`
   branch of the reference implementation with `sigmoid_param_fix_height=True`:
   the sigmoid height is tied to the floor by `a = 1 - b`, `b` is bounded to
   `[0, 0.2]` (equivalently `h` in `[0.8, 1.0]` in the paper's notation), and the
   parameters are fitted by nonlinear least squares with the same initial guess.

`ObservationalScalingPredictor` chains the three stages behind `fit`/`predict`,
so a caller supplies the base benchmark block plus one target column and gets
predictions for rows where the target is hidden.

The target column is mapped onto `metric_range` before fitting and mapped back
afterwards. This is the `minmax_norm` step of the reference implementation,
which takes the metric's own theoretical range as an argument rather than the
observed range, and is a no-op for the accuracy metrics OSL works with. The
distinction matters: the fitted sigmoid rises to the top of that range, so
tying the range to the training rows would cap every prediction at the best
training score and remove the extrapolation OSL is built for.
"""
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

MAX_ITER = 1000
TOL = 1e-4
IMPUTE_COMPONENTS = 1
CAPABILITY_COMPONENTS = 3
SIGMOID_FLOOR_MAX = 0.2
INIT_SLOPE = 3e-2
MAX_FEV = 10000


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def pca_impute(train, test=None, n_components=IMPUTE_COMPONENTS,
               max_iter=MAX_ITER, tol=TOL, bounds=None):
    """Iterative PCA imputation, fitted on `train` and applied to `test`.

    Args:
        train: (n_train, n_features) array with NaN for missing entries.
        test: optional (n_test, n_features) array with NaN for missing entries.
        n_components: rank of the reconstruction used to refill missing cells.
        max_iter: iteration cap, matching the reference implementation.
        tol: absolute tolerance for the convergence check.
        bounds: optional (2, n_features) array of per-feature lower and upper
            limits applied after the inverse scaling. This is the reference
            implementation's `boundary` argument, which its notebooks set to
            [0, 1] for every column because their metrics share one accuracy
            scale. Columns here carry different scales, so the limit is per
            feature rather than one pair for the whole matrix.

    Returns:
        (train_imputed, test_imputed) in the original units; `test_imputed` is
        None when `test` is None.
    """
    train = np.asarray(train, dtype=float)
    scaler = StandardScaler().fit(train)
    train_scaled = scaler.transform(train)
    train_missing = np.isnan(train_scaled)

    imputer = SimpleImputer(strategy='mean').fit(train_scaled)
    train_filled = imputer.transform(train_scaled)

    pca = PCA(n_components=n_components)
    if train_missing.any():
        for _ in range(max_iter):
            reconstruction = pca.inverse_transform(pca.fit_transform(train_filled))
            if np.allclose(train_filled, reconstruction, atol=tol):
                break
            train_filled[train_missing] = reconstruction[train_missing]
    else:
        pca.fit(train_filled)

    train_imputed = scaler.inverse_transform(train_filled)
    if bounds is not None:
        train_imputed = np.clip(train_imputed, bounds[0], bounds[1])
    if test is None:
        return train_imputed, None

    test = np.asarray(test, dtype=float)
    test_scaled = scaler.transform(test)
    test_missing = np.isnan(test_scaled)
    test_filled = imputer.transform(test_scaled)
    if test_missing.any():
        for _ in range(max_iter):
            reconstruction = pca.inverse_transform(pca.transform(test_filled))
            if np.allclose(test_filled, reconstruction, atol=tol):
                break
            test_filled[test_missing] = reconstruction[test_missing]

    test_imputed = scaler.inverse_transform(test_filled)
    if bounds is not None:
        test_imputed = np.clip(test_imputed, bounds[0], bounds[1])
    return train_imputed, test_imputed


def fit_capability_pca(train, n_components=CAPABILITY_COMPONENTS):
    """Fit the capability PCA on complete training rows, without rescaling."""
    pca = PCA(n_components=n_components)
    pca.fit(np.asarray(train, dtype=float))
    return pca


class SigmoidCapabilityRegression:
    """`E = (1 - b) * sigmoid(alpha + beta' S) + b`, fitted by least squares."""

    def __init__(self, floor_max=SIGMOID_FLOOR_MAX):
        self.floor_max = floor_max
        self.alpha = None
        self.beta = None
        self.floor = None

    @staticmethod
    def _design(S):
        S = np.asarray(S, dtype=float)
        return np.column_stack([np.ones(len(S)), S])

    def _model(self, X, *params):
        weights, floor = np.asarray(params[:-1]), params[-1]
        return (1.0 - floor) * _sigmoid(X @ weights) + floor

    def fit(self, S, y):
        X = self._design(S)
        n_weights = X.shape[1]
        p0 = np.concatenate([[0.0], np.full(n_weights - 1, INIT_SLOPE), [0.0]])
        lower = [-np.inf] * n_weights + [0.0]
        upper = [np.inf] * n_weights + [self.floor_max]
        popt, _ = curve_fit(
            self._model, X, np.asarray(y, dtype=float), p0=p0,
            bounds=(lower, upper), maxfev=MAX_FEV,
        )
        self.alpha, self.beta, self.floor = popt[0], popt[1:-1], popt[-1]
        return self

    def predict(self, S):
        params = np.concatenate([[self.alpha], self.beta, [self.floor]])
        return self._model(self._design(S), *params)


class ObservationalScalingPredictor:
    """Predict a hidden target column from a base benchmark block, OSL style.

    Args:
        n_capability_components: number of principal components kept as the
            capability measure, `K` in the paper.
        impute_components: rank used by the iterative PCA imputation.
        metric_range: `(low, high)` theoretical range of the target metric. The
            fitted sigmoid spans this range, so it sets how far above the
            training scores a prediction can reach.
    """

    def __init__(self, n_capability_components=CAPABILITY_COMPONENTS,
                 impute_components=IMPUTE_COMPONENTS, metric_range=(0.0, 1.0)):
        self.n_capability_components = n_capability_components
        self.impute_components = impute_components
        self.metric_low, self.metric_high = float(metric_range[0]), float(metric_range[1])
        if self.metric_high <= self.metric_low:
            raise ValueError('metric_range must be increasing')
        self.pca = None
        self.regression = None

    @property
    def capability_weights(self):
        """Loading of each base benchmark on the fitted capability score."""
        return self.regression.beta @ self.pca.components_

    def fit(self, base_train, target_train, base_test=None):
        """Fit on training rows; `base_test` is imputed under the same transform.

        `base_test` must be supplied here rather than at predict time because
        OSL's imputation is a single fitted transform shared by both splits.
        """
        train_imputed, test_imputed = pca_impute(
            base_train, base_test, n_components=self.impute_components,
        )
        self.pca = fit_capability_pca(
            train_imputed, n_components=self.n_capability_components,
        )
        span = self.metric_high - self.metric_low
        normalized = (np.asarray(target_train, dtype=float) - self.metric_low) / span
        self.regression = SigmoidCapabilityRegression().fit(
            self.pca.transform(train_imputed), normalized,
        )
        return self, test_imputed

    def predict(self, base_imputed):
        scores = self.regression.predict(self.pca.transform(base_imputed))
        return scores * (self.metric_high - self.metric_low) + self.metric_low

    def fit_predict(self, base_train, target_train, base_test):
        _, test_imputed = self.fit(base_train, target_train, base_test)
        return self.predict(test_imputed)
