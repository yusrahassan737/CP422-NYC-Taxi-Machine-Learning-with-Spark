# Databricks notebook source
from pyspark.sql.functions import hour, dayofweek, col, unix_timestamp
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# COMMAND ----------

# Task 2 - Rgressions
#part1
# Load dataset
df = spark.table("default.yellow_tripdata_2015_01")

# Cast timestamps
df = df.withColumn("tpep_pickup_datetime", col("tpep_pickup_datetime").cast("timestamp"))
df = df.withColumn("tpep_dropoff_datetime", col("tpep_dropoff_datetime").cast("timestamp"))

#trip duration in mins
df = df.withColumn(
    "trip_duration",
    ((unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60)
    .cast(DoubleType())
)

#time-based features
df = df.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")).cast(IntegerType()))
df = df.withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")).cast(IntegerType()))

df = df.withColumn("trip_distance", col("trip_distance").cast(DoubleType()))
df = df.withColumn("fare_amount", col("fare_amount").cast(DoubleType()))
df = df.dropna(subset=["trip_duration", "trip_distance", "fare_amount", "pickup_hour", "day_of_week"]
df = df.filter(
    (col("trip_duration") > 0) &
    (col("trip_distance") > 0) &
    (col("fare_amount") > 0)
)

df.printSchema()
df.show(5)

# COMMAND ----------

#clean data
df_clean = df.dropna(subset=["trip_duration", "trip_distance", "fare_amount",
                             "pickup_hour", "day_of_week"]) \
             .filter(
                 (col("trip_duration") > 0) &
                 (col("trip_distance") > 0) &
                 (col("fare_amount") > 0)
             )

feature_cols = ["pickup_hour", "day_of_week", "trip_distance", "trip_duration"]

#train test split
train_df, test_df = df_clean.randomSplit([0.7, 0.3], seed=42)

#validation split
train_inner, val_inner = train_df.randomSplit([0.8, 0.2], seed=42)

#Common transformers
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=False
)

#Hyperparameter search space
regParams        = [0.0, 0.01, 0.1]
elasticNetParams = [0.0, 0.5, 1.0]
maxIters         = [50, 100]

evaluator = RegressionEvaluator(
    labelCol="fare_amount",
    predictionCol="prediction",
    metricName="rmse"
)

best_rmse = float("inf")
best_model = None
best_params = None

#Manual tuning loop
for reg in regParams:
    for enet in elasticNetParams:
        for iters in maxIters:
            lr = LinearRegression(
                featuresCol="scaled_features",
                labelCol="fare_amount",
                regParam=reg,
                elasticNetParam=enet,
                maxIter=iters
            )
            pipeline = Pipeline(stages=[assembler, scaler, lr])
            model = pipeline.fit(train_inner)
            val_preds = model.transform(val_inner)
            rmse = evaluator.evaluate(val_preds)
            print(f"regParam={reg}, elasticNetParam={enet}, maxIter={iters} -> val RMSE={rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = (reg, enet, iters)

print("\nBest params (on validation set):")
print("  regParam       =", best_params[0])
print("  elasticNetParam=", best_params[1])
print("  maxIter        =", best_params[2])
print("Best validation RMSE:", best_rmse)

#eval
test_preds = best_model.transform(test_df)
test_rmse = evaluator.evaluate(test_preds)
r2_evaluator = RegressionEvaluator(
    labelCol="fare_amount",
    predictionCol="prediction",
    metricName="r2"
)
test_r2 = r2_evaluator.evaluate(test_preds)
print("\nPerformance on test set:")
print(f"  Test RMSE: {test_rmse}")
print(f"  Test R^2 : {test_r2}")
test_preds.select(
    "fare_amount", "prediction",
    "pickup_hour", "day_of_week", "trip_distance", "trip_duration"
).show(10, truncate=False)