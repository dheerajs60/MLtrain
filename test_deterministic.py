import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('train.csv')
df = df.dropna(subset=['demand'])

# Drop geohash and timestamp to see if the other features perfectly predict demand
X = df[['day', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather']].copy()
y = df['demand']

cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

# Fill NA
X = X.fillna(0)

rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X, y)

score = rf.score(X, y)
print(f"R2 Score using only basic features: {score}")

