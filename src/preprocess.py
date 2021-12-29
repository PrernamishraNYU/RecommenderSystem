
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import getpass




if __name__ == '__main__':
    netID = getpass.getuser()
    # initialize spark session
    spark = SparkSession.builder.appName('preprocess').getOrCreate()
    # read parquet files
    train = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_train.parquet")
    validate = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_validation.parquet")
    test = spark.read.parquet("hdfs:/user/bm106/pub/MSD/cf_test.parquet")

    temp = train.union(validate)
    whole = temp.union(test)
    
    # construct user hash map
    user_map = whole.select("user_id").distinct()\
        .select("user_id", F.row_number().over(Window.orderBy("user_id")).alias("user_idx"))
    user_map.cache()
    # construct track hash map
    track_map = whole.select("track_id").distinct()\
        .select("track_id", F.row_number().over(Window.orderBy("track_id")).alias("track_idx"))
    track_map.cache()
    # add int index to each
    train = train.join(user_map, "user_id", "inner")
    train = train.join(track_map, "track_id", "inner")
    validate = user_map.join(validate, "user_id", "inner")
    validate = track_map.join(validate, "track_id", "inner")
    test = user_map.join(test, "user_id", "inner")
    test = track_map.join(test, "track_id", "inner")

    # drop columns
    train = train.drop("user_id", "track_id")
    validate = validate.drop("user_id", "track_id")
    test = test.drop("user_id", "track_id")

    # write to disk
    train.repartition("user_idx","track_idx").write.mode("overwrite").parquet(f'hdfs:/user/{netID}/cf_train.parquet')
    validate.repartition("user_idx","track_idx").write.mode("overwrite").parquet(f'hdfs:/user/{netID}/cf_validation.parquet')
    test.repartition("user_idx","track_idx").write.mode("overwrite").parquet(f'hdfs:/user/{netID}/cf_test.parquet')
