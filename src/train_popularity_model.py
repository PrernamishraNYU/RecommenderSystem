import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from lenskit.algorithms.bias import Bias
from lenskit import Recommender
from lenskit import batch

def main(suffix):
	# load data
	train = pd.read_parquet(f'../data/cf_train{suffix}.parquet')
	validation = pd.read_parquet(f'../data/cf_validation{suffix}.parquet')
	test = pd.read_parquet(f'../data/cf_test{suffix}.parquet')
	train.rename(columns={'user_idx':'user', 'count': 'rating', 'track_idx': 'item'}, inplace=True)
	validation.rename(columns={'user_idx':'user', 'count': 'rating', 'track_idx': 'item'}, inplace=True)
	test.rename(columns={'user_idx':'user', 'count': 'rating', 'track_idx': 'item'}, inplace=True)

	#concatenate train and evaluation
	train = pd.concat([train, validation], sort=True)

	# train and evaluate popularity models
	df_merge = train.merge(test, on='user', how='inner')
	users = df_merge.user.unique()
	pred = Bias(items=True, users=False, damping=10**9)
	rec = Recommender.adapt(pred)
	rec.fit(train)
	batch_recs = batch.recommend(rec, users=users, n=500, candidates=None, n_jobs=None)
	batch_recs.to_csv(f'../output/popularity_recommendation{suffix}.csv')

if __name__ == '__main__':
	if len(sys.argv) > 1:
		suffix = '_' + sys.argv[1]
	else:
		suffix = ''
	main(suffix)