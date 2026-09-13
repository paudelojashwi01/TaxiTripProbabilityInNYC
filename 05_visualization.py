"""
Data Visualization
------------------
Loads the trained model and feature-engineered dataset, then produces
the evaluation and exploratory visualizations used in the final report:
ROC curve, actual vs. predicted label distributions, top coefficients,
calendar heatmap of tip %, seasonal box plot, fare/distance scatter
plots, and average tip % by borough.
"""

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

MODEL_PATH = "gs://my-bigdata-project-op/models_final/taxi_tip_model"
TRUSTED_PATH = "gs://my-bigdata-project-op/trusted/taxi_trips_features.parquet"

pipeline = PipelineModel.load(MODEL_PATH)
lr_model = next((stage for stage in pipeline.stages if isinstance(stage, LogisticRegressionModel)), None)
if lr_model is None:
    raise RuntimeError("No LogisticRegressionModel found in pipeline stages")

trusted_df = spark.read.parquet(TRUSTED_PATH)
results = lr_model.transform(trusted_df)

# Recover human-readable feature names from vector metadata
meta = results.schema["features"].metadata["ml_attr"]["attrs"]
n_feats = sum(len(v) for v in meta.values())
feature_names = [None] * n_feats
for v in meta.values():
    for attr in v:
        feature_names[attr["idx"]] = attr["name"]

coeffs = lr_model.coefficients.toArray()
coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coeffs})
print("Top 10 coefficients:\n", coef_df.sort_values("Coefficient", ascending=False).head(10))

# Sample for plotting
sample_sdf = results.sample(False, 0.01, seed=42)
pdf = (
    sample_sdf.withColumn("ProbHighTip", vector_to_array(col("probability"))[1])
    .select(
        col("tip_label").alias("Actual"),
        col("prediction").alias("Predicted"),
        "ProbHighTip", "tip_percent", "fare_amount", "trip_distance",
        col("pickup_date").alias("Date"),
        "ride_season", "ride_day", col("pickup_borough"),
    )
    .toPandas()
)

# ROC curve
fpr, tpr, _ = roc_curve(pdf["Actual"], pdf["ProbHighTip"])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.title("ROC Curve for High-Tip Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Actual vs predicted label distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.countplot(x="Actual", data=pdf, ax=axes[0], palette="viridis")
axes[0].set_title("Actual Tip Label Distribution")
axes[0].set_xlabel("Actual Label (0 = Low, 1 = High)")
sns.countplot(x="Predicted", data=pdf, ax=axes[1], palette="viridis")
axes[1].set_title("Predicted Tip Label Distribution")
axes[1].set_xlabel("Predicted Label (0 = Low, 1 = High)")
plt.tight_layout()
plt.show()

# Calendar heatmap of average tip % by day/month
pdf["Date"] = pd.to_datetime(pdf["Date"])
pdf["month"] = pdf["Date"].dt.month
pdf["day"] = pdf["Date"].dt.day
pivot = pdf.pivot_table(index="day", columns="month", values="tip_percent", aggfunc="mean")

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.5)
plt.title("Calendar Heatmap: Avg Tip % by Day & Month")
plt.xlabel("Month")
plt.ylabel("Day of Month")
plt.tight_layout()
plt.show()

# Seasonal distribution
plt.figure(figsize=(8, 6))
sns.boxplot(data=pdf, x="ride_season", y="tip_percent", order=["Spring", "Summer", "Fall", "Winter"])
plt.title("Tip % Distribution by Season")
plt.xlabel("Season")
plt.ylabel("Tip Percentage")
plt.tight_layout()
plt.show()

# Fare vs tip %
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pdf, x="fare_amount", y="tip_percent", hue="Actual", alpha=0.5)
plt.title("Scatter: Fare Amount vs Tip %")
plt.xlabel("Fare Amount ($)")
plt.ylabel("Tip Percentage")
plt.tight_layout()
plt.show()

# Distance vs tip %
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pdf, x="trip_distance", y="tip_percent", hue="Actual", alpha=0.5)
plt.title("Scatter: Trip Distance vs Tip %")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Tip Percentage")
plt.tight_layout()
plt.show()

# Average tip % by borough
boroughs = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
pdf_borough = pdf[pdf["pickup_borough"].isin(boroughs)]
avg_tip_borough = pdf_borough.groupby("pickup_borough")["tip_percent"].mean().reindex(boroughs)

plt.figure(figsize=(8, 6))
sns.barplot(x=avg_tip_borough.values, y=avg_tip_borough.index, palette="viridis")
plt.title("Average Tip % by Borough (NYC Five Boroughs)")
plt.xlabel("Average Tip Percentage")
plt.ylabel("Borough")
plt.tight_layout()
plt.show()
