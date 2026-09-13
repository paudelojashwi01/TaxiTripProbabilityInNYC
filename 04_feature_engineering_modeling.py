"""
Feature Engineering & Modeling
-------------------------------
Part I:  time-based, cost, location, and airport-ride features; writes
         the enriched dataset to the "trusted" GCS folder.
Part II: builds the Spark ML feature pipeline, trains a Logistic
         Regression model with cross-validated hyperparameter search,
         and evaluates it on the held-out test set.
"""

# ---------------------------------------------------------------------------
# Part I: Feature engineering
# ---------------------------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, month, dayofweek, hour, when, to_date

bucket = "gs://my-bigdataproject"
cleaned_folder = f"{bucket}/cleaned/"
cleaned_filename = f"{cleaned_folder}/*.parquet"

sdf = spark.read.parquet(cleaned_filename)
sample_sdf = sdf.sample(withReplacement=False, fraction=0.01, seed=42)
sdf.printSchema()
pandas_df = sample_sdf.toPandas()

# Quick visual sanity checks on a sample
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=pandas_df, x="passenger_count", palette="Set2")
plt.title("Passenger Count Frequency")

plt.subplot(1, 2, 2)
sns.scatterplot(data=pandas_df, x="trip_distance", y="fare_amount", color="orange")
plt.title("Trip Distance vs Fare Amount")
plt.tight_layout()
plt.show()

cols = ["trip_distance", "total_amount", "trip_duration"]
plt.figure(figsize=(15, 10))
for i, c in enumerate(cols):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=pandas_df, x=c, bins=50, kde=True)
    plt.title(f"Distribution of {c}")
plt.tight_layout()
plt.show()

# Time-based features
sdf = (
    sdf.withColumn("pickup_date", to_date("tpep_pickup_datetime"))
    .withColumn("pickup_month", month("tpep_pickup_datetime"))
    .withColumn("pickup_dayofweek", dayofweek("tpep_pickup_datetime"))
    .withColumn("pickup_hour", hour("tpep_pickup_datetime"))
)

sdf = sdf.withColumn(
    "ride_season",
    when(col("pickup_month").isin(12, 1, 2), "Winter")
    .when(col("pickup_month").between(3, 5), "Spring")
    .when(col("pickup_month").between(6, 8), "Summer")
    .otherwise("Fall"),
)

sdf = sdf.withColumn(
    "ride_day",
    when(col("pickup_dayofweek").isin(1, 7), "Weekend").otherwise("Weekday"),
)

sdf = sdf.withColumn(
    "ride_time_of_day",
    when((col("pickup_hour") >= 5) & (col("pickup_hour") < 12), "Morning")
    .when((col("pickup_hour") >= 12) & (col("pickup_hour") < 17), "Afternoon")
    .when((col("pickup_hour") >= 17) & (col("pickup_hour") < 21), "Evening")
    .otherwise("Late Night"),
)

# Consolidate miscellaneous charges and cross-check against fare
sdf = sdf.withColumn(
    "extra_charge",
    col("extra") + col("mta_tax") + col("tip_amount") + col("tolls_amount")
    + col("improvement_surcharge") + col("congestion_surcharge"),
)
sdf = sdf.withColumn("check_total_amount", col("extra_charge") + col("fare_amount"))

sdf = sdf.fillna({"congestion_surcharge": 0, "airport_fee": 0, "extra_charge": 0})

# Airport-ride flag from RatecodeID (2 = JFK, 3 = Newark)
sdf = sdf.withColumn("airport_ride", when(col("RatecodeID").isin(2, 3), "Yes").otherwise("No"))

# Join borough names via the TLC zone lookup table
lookup_df = spark.read.csv(
    "gs://my-bigdata-project-op/LookUpTable/taxi_zone_lookup.csv", header=True, inferSchema=True
)
pickup_lookup = lookup_df.withColumnRenamed("LocationID", "PULocationID").withColumnRenamed(
    "Borough", "pickup_borough"
)
dropoff_lookup = lookup_df.withColumnRenamed("LocationID", "DOLocationID").withColumnRenamed(
    "Borough", "dropoff_borough"
)

sdf = sdf.join(pickup_lookup.select("PULocationID", "pickup_borough"), on="PULocationID", how="left")
sdf = sdf.join(dropoff_lookup.select("DOLocationID", "dropoff_borough"), on="DOLocationID", how="left")

# Correlation heatmap on numeric features
pdf = sdf.select(
    "fare_amount", "trip_distance", "tip_amount", "extra_charge", "trip_duration", "total_amount", "passenger_count"
).toPandas()
corr = pdf.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

sdf.write.mode("overwrite").parquet(f"{bucket}/trusted")

# ---------------------------------------------------------------------------
# Part II: Modeling
# ---------------------------------------------------------------------------

from pyspark.sql.functions import log1p
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

df = spark.read.parquet("gs://my-bigdataproject/trusted/*.parquet")

df = df.withColumn("tip_percent", (col("tip_amount") / col("fare_amount")) * 100)
df = df.withColumn("tip_label", when(col("tip_percent") >= 15, 1).otherwise(0).cast(IntegerType()))

keep_cols = [
    "pickup_borough", "dropoff_borough", "trip_distance", "RatecodeID", "airport_ride",
    "fare_amount", "ride_season", "ride_day", "ride_time_of_day",
    "extra_charge", "trip_duration", "total_amount",
]
df = df.dropna(subset=keep_cols)

cont_cols = ["trip_distance", "extra_charge", "total_amount", "fare_amount", "trip_duration"]
for c in cont_cols:
    df = df.withColumn(c, col(c).cast("double"))

df = df.withColumn("fare_log", log1p(col("fare_amount")))
df = df.withColumn("duration_log", log1p(col("trip_duration")))

final_cont = ["trip_distance", "total_amount", "extra_charge", "fare_log", "duration_log"]
df = df.dropna(subset=final_cont)

cat_cols = [
    "pickup_borough", "dropoff_borough", "RatecodeID",
    "airport_ride", "ride_season", "ride_day", "ride_time_of_day",
]

indexer = StringIndexer(
    inputCols=cat_cols, outputCols=[c + "_Index" for c in cat_cols], handleInvalid="skip"
)
encoder = OneHotEncoder(
    inputCols=[c + "_Index" for c in cat_cols], outputCols=[c + "_Vec" for c in cat_cols]
)
raw_assembler = VectorAssembler(inputCols=final_cont, outputCol="raw_continuous")
scaler = MinMaxScaler(inputCol="raw_continuous", outputCol="continuousScaled")
final_assembler = VectorAssembler(
    inputCols=[c + "_Vec" for c in cat_cols] + ["continuousScaled"], outputCol="features"
)

feature_pipeline = Pipeline(stages=[indexer, encoder, raw_assembler, scaler, final_assembler])

trusted_path = "gs://my-bigdata-project-op/trusted/taxi_trips_features.parquet"
transformed_df = feature_pipeline.fit(df).transform(df)
transformed_df.write.mode("overwrite").parquet(trusted_path)

sdf = spark.read.parquet(trusted_path)
sdf.groupBy("tip_label").count().orderBy("tip_label").show()

train_df, test_df = sdf.randomSplit([0.7, 0.3], seed=42)

lr = LogisticRegression(featuresCol="features", labelCol="tip_label")
model_pipeline = Pipeline(stages=[lr])

grid = (
    ParamGridBuilder()
    .addGrid(lr.regParam, [0.0, 0.5, 1.0])
    .addGrid(lr.elasticNetParam, [0, 1])
    .build()
)
print("Number of models to be tested:", len(grid))

evaluator = BinaryClassificationEvaluator(labelCol="tip_label", metricName="areaUnderROC")
cv = CrossValidator(estimator=model_pipeline, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3)

all_models = cv.fit(train_df)
bestModel = all_models.bestModel
print("Average metric:", all_models.avgMetrics)

test_results = bestModel.transform(test_df)
test_results.select(
    col("tip_label").alias("Actual"),
    col("prediction").alias("Predicted"),
    "total_amount", "tip_amount", "tip_percent",
).show(10, truncate=False)

cm = (
    test_results.groupBy("tip_label").pivot("prediction").count().fillna(0).sort("tip_label").collect()
)


def calc_metrics(cm):
    tn, fp = cm[0][1], cm[0][2]
    fn, tp = cm[1][1], cm[1][2]
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return accuracy, precision, recall, f1


accuracy, precision, recall, f1 = calc_metrics(cm)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix (label,0.0,1.0):")
print(f"{cm[0][0]}, {cm[0][1]}, {cm[0][2]}")
print(f"{cm[1][0]}, {cm[1][1]}, {cm[1][2]}")

model_path = "gs://my-bigdataproject/models_final/taxi_tip_model"
bestModel.write().overwrite().save(model_path)
