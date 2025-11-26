# task 2 - pt.3

from pyspark.sql.functions import hour, dayofweek, col, unix_timestamp
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.pipeline import PipelineModel


features = ["payment_type", "passenger_count", "trip_distance", "trip_duration"]

df = spark.table("default.yellow_tripdata_2015_01")

df = df.withColumn("tpep_pickup_datetime", col("tpep_pickup_datetime").cast("timestamp"))
df = df.withColumn("tpep_dropoff_datetime", col("tpep_dropoff_datetime").cast("timestamp"))
df = df.withColumn("trip_duration", ((unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60).cast(DoubleType()))
df = df.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")).cast(IntegerType()))
df = df.withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")).cast(IntegerType()))
df = df.withColumn("trip_distance", col("trip_distance").cast(DoubleType()))
df = df.withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

df = df.dropna(subset=["trip_duration", "trip_distance", "fare_amount", "pickup_hour", "day_of_week"]).filter(
    (col("trip_duration") > 0) &
    (col("trip_distance") > 0) &
    (col("fare_amount") > 0)
)

feature_cols = ["pickup_hour", "day_of_week", "trip_distance", "trip_duration"]

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="fare_amount")

rf_pipeline = Pipeline(stages=[assembler, scaler, rf])

rf_param_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [20, 50, 100])
    .addGrid(rf.maxDepth, [5, 10, 15])
    .build()
)

cv_evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")

#(ParamGrid = (ParamGridBuilder())
assembler = VectorAssembler( # I get an error here
    inputCols= features, #features,
    outputCol="features"
)
pipeline = Pipeline(stages = [assembler])

cv = CrossValidator(
    estimator=pipeline, 
    #    estimatorParamMaps=paramGrid,
              
    evaluator=cv_evaluator,
    numFolds=3,
    parallelism=2                      
)
evaluator = RegressionEvaluator(
    labelCol="fare_amount",
    predictionCol="prediction",
    metricName="rmse"
)
train_inner, val_inner = train_df.randomSplit([0.8, 0.2], seed=42)
rf_model = pipeline.fit(train_inner)

train_df = assembler.transform(train_df)
test_df_assembled = assembler.transform(test_df)

lr = LinearRegression(featuresCol="features", labelCol="fare_amount")
lr_model = lr.fit(train_df)
rf_predictions = rf_model.transform(test_df)
predictions = lr_model.transform(test_df)

model = pipeline.fit(train_inner)
best_model = model
test_preds = best_model.transform(test_df)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
rf = RandomForestRegressor(featuresCol="features", labelCol="fare_amount", seed=42)
pipeline = Pipeline(stages=[assembler, rf])
rf_model = pipeline.fit(train_inner)
rf_predictions = rf_model.transform(test_df)
evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
rf_rmse = evaluator.evaluate(rf_predictions)

r2_eval = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2")
rf_r2 = r2_eval.evaluate(rf_predictions)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2:", rf_r2)




#visualizations
# Sample a small subset for faster plotting
sample_preds = rf_predictions.select("fare_amount", "prediction").sample(False, 0.02, seed=42).toPandas()

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(sample_preds["fare_amount"], sample_preds["prediction"], s=5, alpha=0.6)
plt.xlabel("Actual fare")
plt.ylabel("Predicted fare")
plt.title("Actual vs Predicted (Random Forest)")
plt.plot([sample_preds["fare_amount"].min(), sample_preds["fare_amount"].max()],
         [sample_preds["fare_amount"].min(), sample_preds["fare_amount"].max()],
         color='red', linestyle='--')  # optional diagonal reference line
plt.tight_layout()
plt.show()

