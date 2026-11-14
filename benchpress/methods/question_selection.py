"""Representative questions from fitted training-model response profiles."""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances


PROFILE_CLUSTERING = {
    'init': 'k-means++', 'n_init': 3, 'batch_size': 1024,
    'max_iter': 100, 'tol': 0., 'max_no_improvement': 10,
    'reassignment_ratio': 0.01, 'init_size': None,
}


def select_profile_anchors(profiles, n_select, seed):
    """Return sorted, distinct question indices nearest the cluster centers.

    Rows are questions; columns are fitted responses of training models.
    Centers are visited in their fitted order. If their nearest question was
    already chosen, use the nearest unused question, breaking ties by index.
    """
    profiles = np.asarray(profiles, dtype=float)
    if (profiles.ndim != 2 or not profiles.size
            or not np.isfinite(profiles).all()):
        raise ValueError('profiles must be a nonempty finite question-by-model matrix')
    if (not isinstance(n_select, (int, np.integer))
            or not 1 <= n_select <= len(profiles)):
        raise ValueError('n_select must be between one and the number of questions')
    model = MiniBatchKMeans(
        n_clusters=n_select, random_state=seed, **PROFILE_CLUSTERING).fit(profiles)
    distances = pairwise_distances(model.cluster_centers_, profiles, n_jobs=1)
    used = np.zeros(len(profiles), dtype=bool)
    for row in distances:
        row[used] = np.inf
        used[np.argmin(row)] = True
    selected = np.flatnonzero(used)
    if len(selected) != n_select:
        raise AssertionError('representatives do not fill the question budget')
    return selected
