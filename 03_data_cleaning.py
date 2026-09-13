"""
Data Cleaning
-------------
Two-stage cleaning process:

  1. Schema repair (Pandas, per-file) — standardizes dtypes across all
     monthly Parquet files, which otherwise differ file-to-file.
  2. Cleaning rules (PySpark, on a 5-node Dataproc cluster) — filters
     invalid/outlier records and derives trip_duration.

5-node Dataproc cluster used for stage 2:

    gcloud dataproc clusters create cluster-a8fe --enable-component-gateway \
        --region us-central1 --master-machine-type n2-standard-4 \
        --master-boot-disk-type pd-balanced --master-boot-disk-size 100 \
        --num-workers 4 --worker-machine-type n2-standard-4 \
        --worker-boot-disk-type pd-balanced --worker-boot-disk-size 100 \
        --image-version 2.2-debian12 --optional-components JUPYTER \
        --max-idle 3600s --project <PROJECT_ID>
"""

# ---------------------------------------------------------------------------
# Stage 1: Schema repair (Pandas)
# ---------------------------------------------------------------------------

from google.cloud import storage
from io import BytesIO
import pandas as pd
import numpy as np


def main_repair_taxi():
    bucket_name = "projectbucket"
    input_folder = "landing/"
    output_folder = "landing_fixed/"

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = storage_client.list_blobs(bucket_name, prefix=input_folder)
    parquet_blobs = [blob for blob in blobs if blob.name.endswith(".parquet")]

    for blob in parquet_blobs:
        print(f"file {blob.name} with size {blob.size} bytes created on {blob.time_created}")

        df = pd.read_parquet(BytesIO(blob.download_as_bytes()))

        # Standardize dtypes that vary between monthly files
        df["airport_fee"] = df["airport_fee"].astype(np.float64)
        df["RatecodeID"] = df["RatecodeID"].astype(np.float64)
        df["PULocationID"] = df["PULocationID"].astype(np.int64)
        df["DOLocationID"] = df["DOLocationID"].astype(np.int64)
        df["VendorID"] = df["VendorID"].astype(np.int64)
        df["passenger_count"] = df["passenger_count"].astype(np.float64)
        df["improvement_surcharge"] = df["improvement_surcharge"].astype(np.float64)
        df["congestion_surcharge"] = df["congestion_surcharge"].astype(np.float64)

        new_filename = blob.name.replace(input_folder, output_folder)
        print(f"Saving repaired data file to {new_filename}")
        filedata = df.to_parquet(index=False)
        new_blob = bucket.blob(new_filename)
        new_blob.upload_from_string(filedata, content_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Stage 2: Cleaning rules (PySpark)
# ---------------------------------------------------------------------------

from pyspark.sql.functions import col, year, unix_timestamp
import pyspark.sql.functions as F

bucket = "gs://projectbucket"
landing_folder = f"{bucket}/landing_fixed"
cleaning_folder = f"{bucket}/cleaned"

df = spark.read.parquet(landing_folder)

# Keep only a sane year range (adjust to match your acquisition window)
df = df.filter((year(col("tpep_pickup_datetime")) >= 2015) & (year(col("tpep_pickup_datetime")) <= 2021))
df.count()

# Drop columns not useful for modeling
columns_to_drop = ["VendorID", "store_and_fwd_flag"]
df = df.drop(*columns_to_drop)

# A taxi trip must have at least one passenger
df = df.filter((df["passenger_count"].isNotNull()) & (df["passenger_count"] != 0))

MAX_FARE_AMOUNT = 1000
MAX_TIP_AMOUNT = 100
MAX_TOTAL_AMOUNT = 1500
MAX_TOLL_CHARGE = 10
MAX_EXTRA = 5

df = df.filter(
    (F.col("fare_amount") > 0) & (F.col("fare_amount") <= MAX_FARE_AMOUNT)
    & (F.col("tip_amount") >= 0) & (F.col("tip_amount") <= MAX_TIP_AMOUNT)
    & (F.col("total_amount") > 0) & (F.col("total_amount") <= MAX_TOTAL_AMOUNT)
    & (F.col("tolls_amount") > 0) & (F.col("tolls_amount") <= MAX_TOLL_CHARGE)
    & (F.col("extra") > 0) & (F.col("extra") <= MAX_EXTRA)
)

# Surcharges/taxes should not be negative
df = df.filter((F.col("improvement_surcharge") > 0) & (F.col("mta_tax") > 0))

MAX_DISTANCE = 50  # miles — reasonable ceiling for an in-city taxi trip
df = df.filter((F.col("trip_distance") > 0) & (F.col("trip_distance") <= MAX_DISTANCE))

# Derive trip duration (hours) and drop trips longer than 3 hours
df = df.withColumn(
    "trip_duration",
    (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 3600,
)
df = df.filter(col("trip_duration") <= 3)

df.count()
df.printSchema()
df.rdd.getNumPartitions()

cleaned_filename = "gs://projectid/cleaned/cleaned_taxi_data.parquet"
df.write.mode("overwrite").parquet(cleaned_filename)
