# model.py

import pandas as pd
import numpy as np
import joblib
from math import radians, cos, sin, sqrt, atan2
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("Zomato Dataset.csv")

# ----------------------------
# Clean Target Column
# ----------------------------
df["Time_taken (min)"] = (
    df["Time_taken (min)"]
    .astype(str)
    .str.extract(r'(\d+)')[0]
    .astype(float)
)

# ----------------------------
# Numeric Columns
# ----------------------------
numeric_cols = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# ----------------------------
# Time Processing
# ----------------------------
df["Time_Orderd"] = df["Time_Orderd"].astype(str).str.strip()
df["Time_Order_picked"] = df["Time_Order_picked"].astype(str).str.strip()

df["Order_Hour"] = pd.to_datetime(
    df["Time_Orderd"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour

df["Pickup_Hour"] = pd.to_datetime(
    df["Time_Order_picked"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour

df["Order_Hour"] = df["Order_Hour"].fillna(df["Order_Hour"].median())
df["Pickup_Hour"] = df["Pickup_Hour"].fillna(df["Pickup_Hour"].median())

df.drop(["Time_Orderd", "Time_Order_picked"], axis=1, inplace=True)

# ----------------------------
# Haversine Distance
# ----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

df["distance_km"] = df.apply(
    lambda x: haversine(
        x["Restaurant_latitude"],
        x["Restaurant_longitude"],
        x["Delivery_location_latitude"],
        x["Delivery_location_longitude"]
    ),
    axis=1
)

# ----------------------------
# Drop Unnecessary Columns
# ----------------------------
drop_cols = ["ID", "Delivery_person_ID", "Order_Date"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# ----------------------------
# One Hot Encoding
# ----------------------------
df = pd.get_dummies(df, drop_first=True)

# ----------------------------
# Split Features & Target
# ----------------------------
X = df.drop("Time_taken (min)", axis=1)
y = df["Time_taken (min)"]

# ----------------------------
# Train Model (OPTIMIZED)
# ----------------------------
model = RandomForestRegressor(
    n_estimators=50,      # Reduced trees (smaller file)
    max_depth=15,         # Prevent huge trees
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# ----------------------------
# Save Model + Columns (COMPRESSED)
# ----------------------------
joblib.dump(model, "model.joblib", compress=3)
joblib.dump(X.columns.tolist(), "columns.joblib")

print("🔥 Optimized Model saved successfully as model.joblib")
