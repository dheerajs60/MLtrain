import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder
from src.features import build_features
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_df = pd.concat([train, test], ignore_index=True)
all_feat = build_features(all_df, train)

train_feat = all_feat[all_feat['demand'].notna()].copy()
test_feat = all_feat[all_feat['demand'].isna()].copy()

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

target = 'demand'

train_49 = train_feat[train_feat['day'] == 49].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(train_49))

lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31, # Reduced capacity since less data
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

for fold, (train_idx, val_idx) in enumerate(kf.split(train_49)):
    X_train, y_train = train_49.iloc[train_idx][features], train_49.iloc[train_idx][target]
    X_val, y_val = train_49.iloc[val_idx][features], train_49.iloc[val_idx][target]
    
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = np.clip(model_lgb.predict(X_val), 0.0, 1.0)

score_lgb = r2_score(train_49[target], oof_lgb) * 100
print(f"LGBM R2 on Day 49 ONLY: {score_lgb:.4f}")
