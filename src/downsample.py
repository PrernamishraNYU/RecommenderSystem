from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import getpass

def main(spark, netID):
	df = {}
	df['train'] = spark.read.parquet(f'hdfs:/user/{netID}/cf_train.parquet')
	df['validation'] = spark.read.parquet(f'hdfs:/user/{netID}/cf_validation.parquet')
	df['test'] = spark.read.parquet(f'hdfs:/user/{netID}/cf_test.parquet')
	df_all = df['train'].union(df['validation']).union(df['test'])
	users = df_all.select('user_idx').distinct()
	for size, frac in [('small', 0.01), ('medium', 0.05), ('large', 0.25)]:
		user_sample = users.sample(fraction=frac, seed=18)
		for label in ['train', 'validation', 'test']:
			df_sample = user_sample.join(df[label], on='user_idx', how='inner')
			df_sample.write.mode('overwrite').parquet(f'hdfs:/user/{netID}/cf_{label}_{size}.parquet')

if __name__ == '__main__':
	spark = SparkSession.builder.appName('downsample').getOrCreate()
	spark.sparkContext.setLogLevel('ERROR')
	netID = getpass.getuser()
	main(spark, netID)