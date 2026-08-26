import os
import pickle
import tempfile
import unittest

import numpy as np

from benchpress.evaluation_harness import matrix_identity_sha256
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


if __name__ == "__main__":
    unittest.main()
