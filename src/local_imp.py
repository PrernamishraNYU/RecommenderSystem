
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
from pyspark.sql import SparkSession
import itertools as it
import random
import numpy as np

# Load the datasets
train_path = '/Users/prernamishra/Downloads/cf_train_1per.parquet'
val_path = '/Users/prernamishra/Downloads/cf_val_1per.parquet'
test_path = '/Users/prernamishra/Downloads/cf_test_1per.parquet'

spark = SparkSession.builder.appName('preprocess').config('spark.blacklist.enabled', False).getOrCreate()
train = spark.read.parquet(train_path)
val = spark.read.parquet(val_path)
test = spark.read.parquet(test_path)

## Downsample the data
# Pick out user list in training set
user_train = set(row['user_id'] for row in train.select('user_id').distinct().collect())  
# Pick out user list in validation set
user_val = set(row['user_id'] for row in val.select('user_id').distinct().collect())
# Get the previous 1M users
user_prev = list(user_train - user_val)


# Random sampling to get 20%
k = int(0.2 * len(user_prev))
user_prev_filtered = random.sample(user_prev, k)
train = train.where(train.user_id.isin(user_prev_filtered + list(user_val)))


## Create StringIndexer
indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_indexed", handleInvalid = 'skip')
indexer_user_model = indexer_user.fit(train)
indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_indexed", handleInvalid='skip')
indexer_track_model = indexer_track.fit(train)

train = indexer_user_model.transform(train)
train = indexer_track_model.transform(train)

val = indexer_user_model.transform(val)
val = indexer_track_model.transform(val)

test = indexer_user_model.transform(test)
test = indexer_track_model.transform(test)



## ALS model
rank_  = [5, 10, 20]
regParam_ = [0.1, 1, 10]
alpha_ = [1, 5, 10]
param_grid = it.product(rank_, regParam_, alpha_)
user_id = val.select('user_id_indexed').distinct()
true_label = val.select('user_id_indexed', 'track_id_indexed')\
                .groupBy('user_id_indexed')\
                .agg(expr('collect_list(track_id_indexed) as true_item'))



for i in param_grid:
    print('Start Training for {}'.format(i))
    als = ALS(rank = i[0], maxIter=10, regParam=i[1], userCol="user_id_indexed", itemCol="track_id_indexed", ratingCol="count", implicitPrefs=True, \
        alpha=i[2], nonnegative=True, coldStartStrategy="drop")
    model = als.fit(train)
    print('Finish Training for {}'.format(i))

    # Make top 500 recommendations for users in validation test
    res = model.recommendForUserSubset(user_id,500)
    pred_label = res.select('user_id_indexed','recommendations.track_id_indexed')

    pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id_indexed', 'inner') \
                .rdd \
                .map(lambda row: (row[1], row[2]))

    print('Start Evaluating for {}'.format(i))
    metrics = RankingMetrics(pred_true_rdd)
    map_ = metrics.meanAveragePrecision
    ndcg = metrics.ndcgAt(500)
    mpa = metrics.precisionAt(500)
    print(i, 'map score: ', map_, 'ndcg score: ', ndcg, 'mpa score: ', mpa)
print("running model")



# Make predictions on test set
user_id = test.select('user_id_indexed').distinct()
true_label = test.select('user_id_indexed', 'track_id_indexed')\
                .groupBy('user_id_indexed')\
                .agg(expr('collect_list(track_id_indexed) as true_item'))
res = model.recommendForUserSubset(user_id,500)
pred_label = res.select('user_id_indexed','recommendations.track_id_indexed')
pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id_indexed', 'inner') \
                .rdd \
                .map(lambda row: (row[1], row[2]))
metrics = RankingMetrics(pred_true_rdd)
map_ = metrics.meanAveragePrecision
ndcg = metrics.ndcgAt(500)
mpa = metrics.precisionAt(500)

user_factor = model.userFactors
user_vec = user_factor.toPandas()
user_vec = np.array(list(user_vec['features']))
pkl.dump(user_vec, open('user_vec.pkl','wb'))

val_user_id = val.select(val['user_id_indexed'].alias('id')).distinct()
val_tmp = val_user_id.join(user_factor,'id','inner')
val_fea = val_tmp.select('features').toPandas()
val_user_vec = np.array(list(val_fea.features))
pkl.dump(val_user_vec, open('val_user_vec.pkl','wb'))
