import csv
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchpress.methods.predictors import predict_logit_bias_als_scores
from benchpress.methods.transforms import _to_logit, apply_transform
from benchpress.data.score_matrix import ScoreMatrix
from predict import (
	format_score_matrix_report,
	score_matrix_confidence_lookup,
	score_matrix_missing_cells,
)


def write_csv(path, rows):
	with open(path, 'w', newline='') as f:
		csv.writer(f).writerows(rows)


def write_jsonl(path, records):
	with open(path, 'w') as f:
		for record in records:
			f.write(json.dumps(record) + '\n')


class ScoreMatrixTests(unittest.TestCase):
	def write_matrix(self, dir_name):
		csv_path = os.path.join(dir_name, 'scores.csv')
		write_csv(csv_path, [
			['model', 'gpqa_diamond', 'aime_2025', 'chatbot_arena_elo'],
			['my-model-a', '72.0', '55.0', '1310'],
			['my-model-b', '68.5', '', '1288'],
			['my-model-c', '', '61.2', '1400'],
		])
		with open(os.path.join(dir_name, 'scores.meta.json'), 'w') as f:
			json.dump({'chatbot_arena_elo': {'type': 'elo', 'range': [800, 1600]}}, f)
		return csv_path

	def test_csv_loads_metadata_and_missing_cells(self):
		with tempfile.TemporaryDirectory() as dir_name:
			matrix = ScoreMatrix.from_csv(self.write_matrix(dir_name))
			self.assertEqual(matrix.model_ids, ['my-model-a', 'my-model-b', 'my-model-c'])
			self.assertTrue(np.isnan(matrix.values[1, 1]))
			self.assertEqual(matrix.metric['gpqa_diamond']['type'], 'pct')
			self.assertEqual(matrix.metric['chatbot_arena_elo']['type'], 'elo')
			matrix.validate()

	def test_reader_detects_csv_matrix_and_jsonl_records(self):
		with tempfile.TemporaryDirectory() as dir_name:
			csv_matrix = ScoreMatrix.from_file(self.write_matrix(dir_name))
			self.assertEqual(csv_matrix.model_ids[0], 'my-model-a')

			records_path = os.path.join(dir_name, 'records.jsonl')
			write_jsonl(records_path, [
				{'model_id': 'my-model-a', 'benchmark_id': 'gpqa_diamond', 'score': 72.0},
				{'model_id': 'my-model-b', 'benchmark_id': 'aime_2025', 'score': 55.0},
			])
			record_matrix = ScoreMatrix.from_file(records_path)
			self.assertEqual(record_matrix.model_ids, ['my-model-a', 'my-model-b'])
			self.assertEqual(record_matrix.benchmark_ids, ['gpqa_diamond', 'aime_2025'])

	def test_explicit_metric_prediction_path_does_not_forward_metric_to_als(self):
		with tempfile.TemporaryDirectory() as dir_name:
			matrix = ScoreMatrix.from_csv(self.write_matrix(dir_name))
			predictions = predict_logit_bias_als_scores(
				matrix.values,
				metric=matrix.metric,
				benchmark_ids=matrix.benchmark_ids,
			)
			self.assertEqual(predictions.shape, matrix.values.shape)
			self.assertTrue(np.isfinite(predictions[1, 1]))

	def test_non_pct_metadata_skips_logit_transform(self):
		with tempfile.TemporaryDirectory() as dir_name:
			matrix = ScoreMatrix.from_csv(self.write_matrix(dir_name))
			_, _, is_pct, _, _ = apply_transform(matrix.values, _to_logit, True)
			self.assertEqual(is_pct.tolist(), [True, True, False])

	def test_custom_matrix_default_report_is_readable(self):
		with tempfile.TemporaryDirectory() as dir_name:
			csv_path = self.write_matrix(dir_name)
			matrix = ScoreMatrix.from_csv(csv_path)
			predictions = predict_logit_bias_als_scores(
				matrix.values,
				metric=matrix.metric,
				benchmark_ids=matrix.benchmark_ids,
			)
			report = format_score_matrix_report(
				predictions,
				matrix,
				csv_path,
				model_filter='my-model-b',
				only_missing=True,
			)
			self.assertIn('BenchPress estimates for my-model-b', report)
			self.assertIn('Estimated values for scores missing from this matrix', report)
			self.assertIn('aime_2025', report)
			self.assertIn('estimated value', report)

	@patch('benchpress.methods.confidence.predict_confidence_intervals')
	def test_custom_matrix_confidence_report_uses_standard_model(self, predict_confidence):
		with tempfile.TemporaryDirectory() as dir_name:
			csv_path = self.write_matrix(dir_name)
			matrix = ScoreMatrix.from_csv(csv_path)
			predictions = predict_logit_bias_als_scores(
				matrix.values,
				metric=matrix.metric,
				benchmark_ids=matrix.benchmark_ids,
			)
			predict_confidence.return_value = {
				'confidence_level': 0.9,
				'method': 'combined_risk_model',
				'cells': [(1, 1)],
				'predicted': [60.0],
				'uncertainty': [3.0],
				'lower': [55.0],
				'upper': [65.0],
				'trust_probability': [0.8],
				'trust_threshold': 10.0,
			}
			confidence = score_matrix_confidence_lookup(
				matrix,
				predictions,
				[(1, 1)],
			)
			report = format_score_matrix_report(
				predictions,
				matrix,
				csv_path,
				model_filter='my-model-b',
				only_missing=True,
				confidence=confidence,
			)
			self.assertEqual(confidence['__metadata__']['confidence_level'], 0.9)
			self.assertEqual(confidence[(1, 1)]['trust_probability'], 0.8)
			self.assertIn('Confidence:', report)
			self.assertIn('calibrated 90% conformal interval', report)
			self.assertIn('[55.0, 65.0]', report)
			self.assertIn('80%', report)

	def test_custom_matrix_missing_cells_respects_filters(self):
		with tempfile.TemporaryDirectory() as dir_name:
			matrix = ScoreMatrix.from_csv(self.write_matrix(dir_name))
			self.assertEqual(score_matrix_missing_cells(matrix), [(1, 1), (2, 0)])
			self.assertEqual(
				score_matrix_missing_cells(matrix, model_filter='my-model-b'),
				[(1, 1)],
			)
			self.assertEqual(
				score_matrix_missing_cells(matrix, bench_filter='gpqa_diamond'),
				[(2, 0)],
			)


if __name__ == '__main__':
	unittest.main()
