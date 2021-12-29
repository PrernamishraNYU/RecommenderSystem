import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from lenskit.algorithms.bias import Bias
from lenskit import Recommender
from lenskit import batch
from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import RankingMetrics

def main(spark, suffix):
	sc = spark.sparkContext
	batch_recs = pd.read_csv(f'../output/popularity_recommendation{suffix}.csv')
	test = pd.read_parquet(f'../data/cf_test{suffix}.parquet')
	test.rename(columns={'user_idx':'user', 'count': 'rating', 'track_idx': 'item'}, inplace=True)
	prediction_and_labels = []
	users = batch_recs.user.unique()
	for user in users:
		recs = list(batch_recs[batch_recs.user == user].item)
		ground_truths = list(test[test.user == user].item)
		prediction_and_labels.append((recs, ground_truths))
	prediction_and_labels = sc.parallelize(prediction_and_labels)
	metrics = RankingMetrics(prediction_and_labels)
	metric_dict = {}
	metric_dict['mean_avg_pcn'] = metrics.meanAveragePrecision
	k = 50
	metric_dict[f'pcn_at_{k}'] = metrics.precisionAt(k)
	metric_dict[f'ndcg_at_{k}'] = metrics.ndcgAt(k)

	# create metrics csv
	pd.Series(metric_dict).to_csv(f'../output/popularity_metrics{suffix}.csv')

if __name__ == '__main__':
	spark = SparkSession.builder.appName('build_model').getOrCreate()
	if len(sys.argv) > 1:
		suffix = '_' + sys.argv[1]
	else:
		suffix = ''
	main(spark, suffix)