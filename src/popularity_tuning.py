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

	# load data
	train = pd.read_parquet(f'../data/cf_train{suffix}.parquet')
	validation = pd.read_parquet(f'../data/cf_validation{suffix}.parquet')
	train.rename(columns={'user_idx':'user', 'count': 'rating', 'track_idx': 'item'}, inplace=True)
	validation.rename(columns={'user_idx':'user', 'count': 'rating', 'track_idx': 'item'}, inplace=True)

	# train and evaluate popularity models
	top_exp = 10
	params = [10**n for n in range(top_exp)]
	df_merge = train.merge(validation, on='user', how='inner')
	users = df_merge.user.unique()
	mean_avg_pcn = []
	for param in params:
		pred = Bias(items=True, users=False, damping=param)
		rec = Recommender.adapt(pred)
		rec.fit(train)
		batch_recs = batch.recommend(rec, users=users, n=500, candidates=None, n_jobs=None)
		prediction_and_labels = []
		for user in users:
			recs = list(batch_recs[batch_recs.user == user].item)
			ground_truths = list(validation[validation.user == user].item)
			prediction_and_labels.append((recs, ground_truths))
		prediction_and_labels = sc.parallelize(prediction_and_labels)
		metrics = RankingMetrics(prediction_and_labels)
		mean_avg_pcn.append(metrics.meanAveragePrecision)
	opt_param_idx = np.argmax(mean_avg_pcn)
	opt_param = params[opt_param_idx]
	log_10_param = range(top_exp)

	# create plot
	plt.plot(log_10_param, mean_avg_pcn)
	plt.xlabel('Log10(Damping Parameter)')
	plt.ylabel('Mean Average Precision')
	plt.title('Mean Average Precision vs Damping Parameter')
	plt.savefig(f'../output/tuned_damping{suffix}.png')

	# create mean average precision csv
	pd.Series(mean_avg_pcn).to_csv(f'../output/mean_avg_pcn{suffix}.csv')

if __name__ == '__main__':
	spark = SparkSession.builder.appName('build_model').getOrCreate()
	if len(sys.argv) > 1:
		suffix = '_' + sys.argv[1]
	else:
		suffix = ''
	main(spark, suffix)