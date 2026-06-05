import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from src.features import build_features

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_df = pd.concat([train, test], ignore_index=True)
all_feat = build_features(all_df, train)

train_feat = all_feat[all_feat['demand'].notna()].copy()

cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks', 'geohash', 'geohash_prefix4', 'geohash_prefix5']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_feat[cat_cols] = encoder.fit_transform(train_feat[cat_cols].astype(str))

features = [
    'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
    'hour', 'minute', 'time_minutes', 'sin_time', 'cos_time', 'latitude', 'longitude',
    'geohash', 'geohash_prefix4', 'geohash_prefix5',
    'demand_at_2_00', 'demand_at_1_45', 'demand_at_1_30', 'demand_at_1_15', 'demand_at_1_00', 'mean_demand_0_to_2',
    'mean_demand_0_to_2_prefix5', 'mean_demand_0_to_2_prefix4',
    'demand_day48_t', 'demand_day48_t_minus_15', 'demand_day48_t_minus_30', 'demand_day48_t_minus_60',
    'demand_day48_t_plus_15', 'demand_day48_t_plus_30', 'demand_day48_t_plus_60',
    'demand_day48_rolling_mean_30', 'demand_day48_rolling_mean_60',
    'day48_overall_mean', 'day48_overall_std', 'day48_overall_max', 'day48_overall_min',
    'day48_prefix5_mean', 'day48_prefix5_std', 'day48_prefix5_max', 'day48_prefix5_min',
    'day48_prefix4_mean', 'day48_prefix4_std', 'day48_prefix4_max', 'day48_prefix4_min',
    'demand_day48_prefix5_t', 'demand_day48_prefix4_t', 'city_profile_day48'
]

train_49 = train_feat[train_feat['day'] == 49]

X = train_49[features]
y = train_49['demand']

model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
model.fit(X, y)

importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
print(importance.sort_values('importance', ascending=False).head(20))
