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

# --- Feature Improvements ---

# 1. Ratio of today's early morning demand to yesterday's overall early morning demand
# We already have mean_demand_0_to_2 (today's morning). We need yesterday's morning.
# Let's approximate yesterday's morning using demand_day48_t_minus/plus... actually we can just average demand_day48_t where time_minutes <= 120
# For simplicity, let's just use day48_overall_mean as the denominator for the ratio
all_feat['morning_to_day48_ratio'] = all_feat['mean_demand_0_to_2'] / (all_feat['day48_overall_mean'] + 1e-6)

# 2. Temperature squared (might capture non-linear effects better)
all_feat['Temperature_sq'] = all_feat['Temperature'] ** 2

# 3. Ratio of prefix4 morning to day48 prefix4
all_feat['morning_prefix4_ratio'] = all_feat['mean_demand_0_to_2_prefix4'] / (all_feat['day48_prefix4_mean'] + 1e-6)

# --------------------------

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
    'demand_day48_prefix5_t', 'demand_day48_prefix4_t', 'city_profile_day48',
    'morning_to_day48_ratio', 'Temperature_sq', 'morning_prefix4_ratio'
]

target = 'demand'

train_48 = train_feat[train_feat['day'] == 48].copy()
train_49 = train_feat[train_feat['day'] == 49].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_lgb = np.zeros(len(train_49))
oof_lgb_log = np.zeros(len(train_49))

lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1
}

print("Running CV...")
for fold, (train_idx, val_idx) in enumerate(kf.split(train_49)):
    fold_train_49 = train_49.iloc[train_idx]
    fold_train = pd.concat([train_48, fold_train_49], ignore_index=True)
    
    X_train, y_train = fold_train[features], fold_train[target]
    fold_val = train_49.iloc[val_idx]
    X_val, y_val = fold_val[features], fold_val[target]
    
    # Standard Model
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = np.clip(model_lgb.predict(X_val), 0.0, 1.0)
    
    # Log1p Model
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    model_lgb_log = lgb.LGBMRegressor(**lgb_params)
    model_lgb_log.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb_log[val_idx] = np.clip(np.expm1(model_lgb_log.predict(X_val)), 0.0, 1.0)

y_49 = train_49[target].values
score_lgb = r2_score(y_49, oof_lgb) * 100
score_lgb_log = r2_score(y_49, oof_lgb_log) * 100

print(f"Standard LGBM R2: {score_lgb:.4f}")
print(f"Log1p Transformed LGBM R2: {score_lgb_log:.4f}")
