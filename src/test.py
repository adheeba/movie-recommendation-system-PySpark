from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import explode


# Create a SparkSession
spark = SparkSession.builder.appName("Movie Recommendation App") \
                            .config("spark.executor.memory", "8g") \
                            .getOrCreate()
#spark.conf.set("spark.sql.shuffle.partitions", 200)

# Load data
ratings_all = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("./data/ratings.csv")

# Process data
ratings = ratings_all.withColumn("userId", col("userId").cast("int")) \
                     .withColumn("movieId", col("movieId").cast("int")) \
                     .withColumn("rating", col("rating").cast("float")) \
                     .drop('timestamp')
#ratings = ratings.limit(200)
# Split data
(train, test) = ratings.randomSplit([0.80, 0.20], seed=1234)

# ALS model
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", 
          nonnegative=True, implicitPrefs=False, coldStartStrategy="drop")
'''
# Hyperparameter tuning
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 50]) \
            .addGrid(als.maxIter, [5, 50]) \
            .addGrid(als.regParam, [.01, .05]) \
            .build()
'''
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# CrossValidator
#cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

# Fit model
model = als.fit(train)
#best_model = model.bestModel
# Make predictions
test_predictions = model.transform(test)
#test_predictions.show()

# Calculate RMSE
rmse = evaluator.evaluate(test_predictions)
print(f"Root Mean Square Error (RMSE): {rmse}")

#  Generate Recommendations
user_recs = model.recommendForAllUsers(10)
movie_recs = model.recommendForAllItems(10)

recommendationsDF = (user_recs
  .select("userId", explode("recommendations")
  .alias("recommendation"))
  .select("userId", "recommendation.*")
)
recommendationsDF.show(5)
recommendationsDF.write.csv("./data/recommendations", header=True, mode='overwrite')

# Save model
model.write().overwrite().save("./model")
#recommendationsDF.show(5)
test.show(5)
# Stop Spark
spark.stop()
