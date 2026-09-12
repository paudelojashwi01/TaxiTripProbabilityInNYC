# TaxiTripProbabilityInNYC

An end-to-end PySpark pipeline that predicts whether an NYC Yellow Taxi passenger will tip ≥15% of the fare, built on Google Cloud (GCS + Dataproc). The project covers data acquisition, cleaning, feature engineering, modeling, and visualization for ~3M+ trip records.

## Project Overview ## 
Using only trip-level attributes (distance, duration, time, location, and surcharges), this project frames tipping behavior as a binary classification problem and trains a Logistic Regression model to distinguish "good tippers" from other riders. Beyond prediction, the analysis surfaces the key drivers of tipping which are fare size, borough, and rate code to support driver incentive design, customer targeting, and dynamic pricing strategies.

## Dataset ## 
[TLC Trip Record Data] (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), focused on Yellow Taxi trips. Key fields: pickup/dropoff timestamps, trip distance, passenger count, fare components, payment type, and pickup/dropoff zone IDs.

## Architecture ## 
TLC Cloudfront (Parquet) → GCS "landing" bucket → Dataproc (single-node, EDA)
    → Dataproc (5-node, PySpark cleaning) → GCS "cleaned"
    → Feature Engineering (Spark ML Pipeline) → GCS "trusted"
    → Logistic Regression + CrossValidator → GCS "models"
    → Evaluation & Visualization (Pandas/Matplotlib/Seaborn)


