"""
Exploratory Data Analysis
-------------------------
Run on a single-node Google Cloud Dataproc cluster:

    gcloud dataproc clusters create cluster-1234 --enable-component-gateway \
        --region us-central1 --single-node --master-machine-type n2-standard-16 \
        --master-boot-disk-type pd-balanced --master-boot-disk-size 100 \
        --image-version 2.2-debian12 --optional-components JUPYTER \
        --max-idle 3600s --project <PROJECT_ID>

Reads all Parquet files from the "landing/" folder in GCS into a single
Pandas DataFrame and inspects schema, duplicates, missing values, and
distributions for each key column.
"""

from google.cloud import storage
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bucket_name = "projectname"
storage_client = storage.Client()

# Load all monthly Parquet files from the landing folder
blobs = storage_client.list_blobs(bucket_name, prefix="landing/")
parquet_blobs = [blob for blob in blobs if blob.name.endswith(".parquet")]

df_list = []
for blob in parquet_blobs:
    print(f"Reading {blob.name} ({blob.size} bytes)")
    df_temp = pd.read_parquet(BytesIO(blob.download_as_bytes()))
    df_list.append(df_temp)

df = pd.concat(df_list, ignore_index=True)
print(f"\nThis data contains {df.shape[0]} rows and {df.shape[1]} columns.")
df.dtypes
df.head().T


def check_duplicates(df):
    if df.duplicated().sum() == 0:
        print("This data has no duplicated records.")
    else:
        print(f"This data has {df.duplicated().sum()} duplicated records.")


check_duplicates(df)


def check_missing_values(df):
    if df.isnull().sum().sum() == 0:
        print("This data has no missing values.")
    else:
        print("\nThis data has missing values distributed as follows:")
        return df.isnull().sum()


check_missing_values(df)

# Vendor distribution
print(df["VendorID"].value_counts())

# Year sanity check on pickup/dropoff timestamps
df["pickup_year"] = df["tpep_pickup_datetime"].dt.year
df["dropoff_year"] = df["tpep_dropoff_datetime"].dt.year
print("\nNumber of records for each year:")
print(df["pickup_year"].value_counts())
print(df["dropoff_year"].value_counts())

# Trip duration sanity check (surfaces negative/extreme durations)
df["trip_duration"] = (
    df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60.0
print(df["trip_duration"].describe())

# Passenger count
print(df["passenger_count"].describe())
print(df["passenger_count"].value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(x="passenger_count", data=df, color="#FCE883")
plt.title("Passenger Count Distribution (0 to 9)")
plt.xlabel("Passenger Count")
plt.ylabel("Frequency")
plt.xticks(range(0, 10))
plt.show()

# Trip distance (zero-distance trips)
print(df["trip_distance"].describe())
zero_distance_trips = df[df["trip_distance"] == 0].shape[0]
non_zero_distance_trips = df.shape[0] - zero_distance_trips
print(f"Zero distance trips: {zero_distance_trips}")

plt.figure(figsize=(8, 6))
df["trip_distance"].apply(
    lambda x: "Zero Distance" if x == 0 else "Non-Zero Distance"
).value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
    colors=["#FADA5E", "salmon"],
    labels=["Non-Zero Distance", "Zero Distance"],
)
plt.title("Zero vs Non-Zero Trip Distances")
plt.ylabel("")
plt.show()

# Rate code and payment type
print(df["RatecodeID"].value_counts())
print(df["payment_type"].value_counts())

payment_type_labels = {
    0: "0: Flex Fare Trip",
    1: "1: Credit Card",
    2: "2: Cash",
    3: "3: No Charge",
    4: "4: Dispute",
    5: "5: Unknown",
}
df["payment_type_label"] = df["payment_type"].map(payment_type_labels)

plt.figure(figsize=(8, 6))
sns.barplot(
    x=df["payment_type_label"].value_counts().index,
    y=df["payment_type_label"].value_counts().values,
    palette="pastel",
)
plt.title("Payment Type Distribution", fontsize=16)
plt.xlabel("Payment Type", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.show()

# Fare / total / tip sanity checks
print("\nDescriptive Statistics for fare_amount:")
print(df["fare_amount"].describe())
print(f"\nNumber of trips with negative or zero fare amounts: {(df['fare_amount'] <= 0).sum()}")

print("\nDescriptive Statistics for total_amount:")
print(df["total_amount"].describe())
print(f"\nNumber of trips with negative or zero total amounts: {(df['total_amount'] <= 0).sum()}")

print("\nDescriptive Statistics for tip_amount:")
print(df["tip_amount"].describe())

extra_charges = [
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
]
print("\nSummary Statistics for Extra Charges\n")
print(df[extra_charges].describe())

# Pickup/dropoff zone popularity
print("Number of unique PULocationID values:", df["PULocationID"].nunique())
print("Number of unique DOLocationID values:", df["DOLocationID"].nunique())
print(df["PULocationID"].value_counts().head())
print(df["DOLocationID"].value_counts().head())
