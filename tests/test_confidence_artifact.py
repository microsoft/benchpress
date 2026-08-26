import os
import pickle
import tempfile
import unittest

import numpy as np

from benchpress.evaluation_harness import (
    benchmark_metric_identity_sha256,
    matrix_identity_sha256,
)
from benchpress.methods.confidence import (
    load_or_train_default_confidence_calibrator,
)


class ConfidenceArtifactTests(unittest.TestCase):
    def test_matrix_identity_tracks_values_and_missingness(self):
        matrix = np.asarray([[1.0, np.nan], [2.0, 3.0]])
        identity = matrix_identity_sha256(matrix)
        self.assertEqual(identity, matrix_identity_sha256(matrix.copy()))

        changed_value = matrix.copy()
        changed_value[0, 0] = 1.1
        self.assertNotEqual(identity, matrix_identity_sha256(changed_value))

        changed_mask = matrix.copy()
        changed_mask[0, 1] = 0.0
        self.assertNotEqual(identity, matrix_identity_sha256(changed_mask))

    def test_stale_default_calibrator_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = os.path.join(directory, "calibrator.pkl")
            with open(artifact_path, "wb") as file:
                pickle.dump({
                    "version": 1,
                    "matrix_shape": [84, 133],
                    "calibrators": {},
                }, file)
            with self.assertRaisesRegex(
                    ValueError, "does not match the canonical score matrix"):
                load_or_train_default_confidence_calibrator(
                    artifact_path=artifact_path,
                    train_if_missing=False,
                )

    def test_benchmark_metric_identity_tracks_ordered_semantics(self):
        metric = {
            "accuracy": {
                "type": "pct",
                "range": [0, 100],
                "higher_is_better": True,
            },
            "edit_distance": {
                "type": "normalized_edit_distance",
                "range": [0, 1],
                "higher_is_better": False,
            },
        }
        identity = benchmark_metric_identity_sha256(
            metric, ["accuracy", "edit_distance"])
        self.assertEqual(
            identity,
            benchmark_metric_identity_sha256(
                metric.copy(), ["accuracy", "edit_distance"]),
        )

        changed = {key: dict(value) for key, value in metric.items()}
        changed["edit_distance"]["range"] = [0, 100]
        self.assertNotEqual(
            identity,
            benchmark_metric_identity_sha256(
                changed, ["accuracy", "edit_distance"]),
        )
        self.assertNotEqual(
            identity,
            benchmark_metric_identity_sha256(
                metric, ["edit_distance", "accuracy"]),
        )


if __name__ == "__main__":
    unittest.main()
