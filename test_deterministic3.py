import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('train.csv')
df = df.dropna(subset=['demand'])

df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
df['time_minutes'] = df['hour'] * 60 + df['minute']

X = df[['RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather', 'time_minutes', 'geohash']].copy()
y = df['demand']

cat_cols = ['RoadType', 'LargeVehicles', 'Landmarks', 'Weather', 'geohash']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))

X = X.fillna(0)

rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X, y)

score = rf.score(X, y)
print(f"R2 Score WITHOUT 'day': {score}")

