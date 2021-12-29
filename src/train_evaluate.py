import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
import sys
import getpass
from time import time


def ranking_cf(df, top_k):
    """
    :param df: input dataframe
    :param top_k: top k tracks for each user
    :return: df with cols (user, tracks) where tracks is a list of top k highest count tracks for user i
    """

    window = Window.partitionBy(df['user_idx']).orderBy(df['count'].desc())

    rec_df = df.select('*', F.rank().over(window).alias('rank')).filter(F.col('rank') <= top_k)\
        .orderBy('user_idx', 'rank').groupby('user_idx').agg(F.collect_list('track_idx').alias('tracks'))

    return rec_df


def main():
    # indicate which file to read
    file_size = sys.argv[3]
    # get user
    netID = getpass.getuser()
    # initialize spark session
    spark = SparkSession.builder.appName('preprocess').getOrCreate()
    # read parquet files
    if file_size == "full":
        train = spark.read.parquet(f"hdfs:/user/{netID}/cf_train.parquet")
        test = spark.read.parquet(f"hdfs:/user/{netID}/cf_test.parquet")
    else:
        train = spark.read.parquet(f"hdfs:/user/{netID}/cf_train_{file_size}.parquet")
        test = spark.read.parquet(f"hdfs:/user/{netID}/cf_test_{file_size}.parquet")
    test.cache()

    # setting parameters
    rank = int(sys.argv[1])
    reg = float(sys.argv[2])
    maxiter = 20
    print(f"training ALS with rank = {rank} and regularization parameter = {reg}")
    # training
    als = ALS(rank=rank, regParam=reg, maxIter=maxiter, numUserBlocks=20, numItemBlocks=20, implicitPrefs=True,
              userCol="user_idx", itemCol="track_idx", ratingCol="count")
    model = als.fit(train)

    # evaluate
    true_rank = ranking_cf(test, 500)
    users_in_test = test.select("user_idx").distinct()
    predict_rank = model.recommendForUserSubset(users_in_test, 500)
    predict_rank = predict_rank.join(true_rank, "user_idx", "inner").select('recommendations.track_idx', 'tracks')

    # to rdd
    labelrdd = predict_rank.rdd.map(lambda x: (x[0], x[1]),
                                    preservesPartitioning=True)
    # calculate MAP
    metric = RankingMetrics(labelrdd)
    MAP = metric.meanAveragePrecision
    precision = metric.precisionAt(500)
    ndcg = metric.ndcgAt(500)

    print(f"current ALS model mean average precision on test data {MAP}")
    print(f"current ALS model with precision at 500 {precision}")
    print(f"current ALS model with ndcg at 500 {ndcg}")



if __name__ == '__main__':
    start = time()
    main()
    print(f"total time spend on training and evaluation is {time() - start}")
