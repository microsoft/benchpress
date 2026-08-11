import csv
import json
import os
import tempfile
import unittest

import numpy as np

from benchpress.methods.predictors import predict_logit_bias_als_scores
from benchpress.methods.transforms import _to_logit, apply_transform
from benchpress.data.score_matrix import ScoreMatrix
from predict import (
	evaluate_score_matrix_holdout,
	format_score_matrix_report,
	load_or_compute_score_matrix_holdout,
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
			self.assertIn('Prediction results for my-model-b', report)
			self.assertIn('Showing missing cells only', report)
			self.assertIn('aime_2025', report)
			self.assertIn('predicted', report)

	def test_custom_matrix_confidence_reports_holdout_error(self):
		with tempfile.TemporaryDirectory() as dir_name:
			csv_path = self.write_matrix(dir_name)
			matrix = ScoreMatrix.from_csv(csv_path)
			predictions = predict_logit_bias_als_scores(
				matrix.values,
				metric=matrix.metric,
				benchmark_ids=matrix.benchmark_ids,
			)
			validation = evaluate_score_matrix_holdout(matrix)
			report = format_score_matrix_report(
				predictions,
				matrix,
				csv_path,
				model_filter='my-model-b',
				only_missing=True,
				validation=validation,
			)
			self.assertGreater(validation['summary']['n_eval'], 0)
			self.assertIsNotNone(validation['summary']['medae'])
			self.assertIsNotNone(validation['summary']['medape'])
			self.assertIn('Holdout validation on observed cells', report)
			self.assertIn('test MedAE', report)
			self.assertIn('test MedAPE', report)
			self.assertIn('interval_90', report)

	def test_custom_matrix_holdout_uses_matrix_folder_cache(self):
		with tempfile.TemporaryDirectory() as dir_name:
			csv_path = self.write_matrix(dir_name)
			matrix = ScoreMatrix.from_csv(csv_path)
			validation = load_or_compute_score_matrix_holdout(matrix, csv_path)
			cache_dir = os.path.join(dir_name, '__benchpress_cache__')
			cache_files = os.listdir(cache_dir)
			self.assertEqual(len(cache_files), 1)
			self.assertGreater(validation['summary']['n_eval'], 0)

			cache_path = os.path.join(cache_dir, cache_files[0])
			with open(cache_path) as f:
				cached_validation = json.load(f)
			cached_validation['summary']['n_eval'] = 123
			with open(cache_path, 'w') as f:
				json.dump(cached_validation, f)

			loaded_validation = load_or_compute_score_matrix_holdout(matrix, csv_path)
			self.assertEqual(loaded_validation['summary']['n_eval'], 123)


if __name__ == '__main__':
	unittest.main()
