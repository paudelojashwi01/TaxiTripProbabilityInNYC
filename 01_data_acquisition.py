"""
Data Acquisition
----------------
Downloads NYC Yellow Taxi trip data (Parquet) from the TLC Cloudfront
endpoint and stages it in a Google Cloud Storage bucket.

GCS bucket setup and upload are shown as gcloud CLI commands below the
Python download script for reference.
"""

from urllib.request import urlretrieve

years_list = ["2019", "2020", "2021", "2022"]
months_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

for year in years_list:
    for month in months_list:
        filename = f"yellow_tripdata_{year}-{month}.parquet"
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
        urlretrieve(url, filename)

# ---------------------------------------------------------------------------
# GCS setup (run via gcloud CLI, not Python)
# ---------------------------------------------------------------------------
#
# 1. Create the bucket:
#    gcloud storage buckets create gs://my-bigdata-project-op --project=<PROJECT_ID> \
#        --default-storage-class=STANDARD --location=us-central1 --uniform-bucket-level-access
#
# 2. Upload the downloaded Parquet files to the landing directory:
#    gcloud storage cp yellow_tripdata_*.parquet gs://my-bigdata-project-op/landing/
#
# 3. Initialize directories for each pipeline stage:
#    gcloud storage cp /dev/null gs://my-bigdata-project-op/cleaned
#    gcloud storage cp /dev/null gs://my-bigdata-project-op/trusted
#    gcloud storage cp /dev/null gs://my-bigdata-project-op/code
#    gcloud storage cp /dev/null gs://my-bigdata-project-op/models
