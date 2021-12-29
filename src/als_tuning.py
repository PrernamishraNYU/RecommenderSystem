import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
import getpass

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
    netID = getpass.getuser()
    # initialize spark session
    spark = SparkSession.builder.appName('preprocess').config('spark.blacklist.enabled', False).getOrCreate()
    # read parquet files
    train = spark.read.parquet(f"hdfs:/user/{netID}/cf_train_large.parquet")
    test = spark.read.parquet(f"hdfs:/user/{netID}/cf_validation_large.parquet")
    test.cache()

    # setting parameters
    als = ALS()


    #max iter suggested 20-30 range
    #rank somewhere under 500
    #reg param scaled up until starts to hurt

    als = ALS(maxIter=25, numUserBlocks=20, numItemBlocks=20, implicitPrefs=True,
              userCol="user_idx", itemCol="track_idx", ratingCol="count")
    tolerance = 0.03
    ranks = [50,100]
    regParams = [1]
    errors = [[0]*len(ranks)]*len(regParams)
    models = [[0]*len(ranks)]*len(regParams)
    err = 0
    max_MAP = 0
    best_rank = -1

    # get distinct users from testing data
    user_in_test = test.select("user_idx").distinct()
    true_rank = ranking_cf(test, 500)
    i = 0
    for regParam in regParams:
        j = 0
        for rank in ranks:
            als.setParams(rank = rank, regParam = regParam)
            model = als.fit(train)


            # evaluate
            predict_rank = model.recommendForUserSubset(user_in_test, 500)
            predict_rank = predict_rank.join(true_rank, "user_idx", "inner").select('recommendations.track_idx', 'tracks')

            # to rdd
            labelrdd = predict_rank.rdd.map(lambda x: (x[0], x[1]),
                                            preservesPartitioning=True)
            # calculate MAP
            MAP = RankingMetrics(labelrdd).meanAveragePrecision
            errors[i][j] = MAP
            models[i][j] = model
            print('For rank %s, regularization parameter %s the MAP is %s' % (rank, regParam, MAP))
            if MAP > max_MAP:
                min_error = MAP
                best_params = [i,j]
            j += 1
        i += 1
    #print(f"current ALS model mean average precision on test data {MAP}")
    als.setRegParam(regParams[best_params[0]])
    als.setRank(ranks[best_params[1]])
    print('The best model was trained with regularization parameter %s' % regParams[best_params[0]])
    print ('The best model was trained with rank %s' % ranks[best_params[1]])
    my_model = models[best_params[0]][best_params[1]]

if __name__ == '__main__':
    main()
