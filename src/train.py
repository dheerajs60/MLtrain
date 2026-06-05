import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder
from src.features import build_features
import warnings
warnings.filterwarnings('ignore')

def train_and_evaluate():
    print("Step 1: Loading Datasets...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    print("Step 2: Preprocessing and Feature Engineering...")
    # Combine for consistent encoding
    all_df = pd.concat([train, test], ignore_index=True)
    all_feat = build_features(all_df, train)
    
    train_feat = all_feat[all_feat['demand'].notna()].copy()
    test_feat = all_feat[all_feat['demand'].isna()].copy()
    
    cat_cols = ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks', 'geohash', 'geohash_prefix4', 'geohash_prefix5']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Fit encoder on train and transform both
    train_feat[cat_cols] = encoder.fit_transform(train_feat[cat_cols].astype(str))
    test_feat[cat_cols] = encoder.transform(test_feat[cat_cols].astype(str))
    
    features = [
        'day', 'RoadType', 'NumberofLanes', 'LargeVehicles', 'Landmarks', 'Temperature', 'Weather',
        'hour', 'minute', 'time_minutes', 'latitude', 'longitude',
        'geohash', 'geohash_prefix4', 'geohash_prefix5'
    ]
    
    target = 'demand'
    
    print(f"Train shape: {train_feat.shape}")
    print(f"Test shape: {test_feat.shape}")
    
    print("\nStep 3: Starting Cross-Validation & Training...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(train_feat))
    oof_xgb = np.zeros(len(train_feat))
    oof_cb = np.zeros(len(train_feat))
    
    test_lgb = np.zeros(len(test_feat))
    test_xgb = np.zeros(len(test_feat))
    test_cb = np.zeros(len(test_feat))
    
    # Deeper trees since it's a deterministic tabular dataset!
    lgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': -1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42,
        'verbose': -1
    }
    
    xgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'max_depth': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42,
        'verbosity': 0
    }
    
    cb_params = {
        'iterations': 2000,
        'learning_rate': 0.05,
        'depth': 10,
        'random_seed': 42,
        'verbose': 0
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_feat)):
        print(f"\n--- Training Fold {fold} ---")
        
        X_train, y_train = train_feat.iloc[train_idx][features], train_feat.iloc[train_idx][target]
        X_val, y_val = train_feat.iloc[val_idx][features], train_feat.iloc[val_idx][target]
        
        # LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[val_idx] = np.clip(model_lgb.predict(X_val), 0.0, 1.0)
        test_lgb += np.clip(model_lgb.predict(test_feat[features]), 0.0, 1.0) / kf.n_splits
        
        # XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = np.clip(model_xgb.predict(X_val), 0.0, 1.0)
        test_xgb += np.clip(model_xgb.predict(test_feat[features]), 0.0, 1.0) / kf.n_splits
        
        # CatBoost
        model_cb = CatBoostRegressor(**cb_params)
        model_cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        oof_cb[val_idx] = np.clip(model_cb.predict(X_val), 0.0, 1.0)
        test_cb += np.clip(model_cb.predict(test_feat[features]), 0.0, 1.0) / kf.n_splits
        
        # Calculate fold metrics
        r2_lgb = r2_score(y_val, oof_lgb[val_idx]) * 100
        r2_xgb = r2_score(y_val, oof_xgb[val_idx]) * 100
        r2_cb = r2_score(y_val, oof_cb[val_idx]) * 100
        print(f"Fold {fold} R2 Scores -> LGBM: {r2_lgb:.4f} | XGB: {r2_xgb:.4f} | CatBoost: {r2_cb:.4f}")

    print("\nStep 4: Evaluating Out-of-Fold (OOF) Metrics...")
    score_lgb = r2_score(train_feat[target], oof_lgb) * 100
    score_xgb = r2_score(train_feat[target], oof_xgb) * 100
    score_cb = r2_score(train_feat[target], oof_cb) * 100
    
    oof_ensemble = (oof_lgb + oof_xgb + oof_cb) / 3.0
    score_ensemble = r2_score(train_feat[target], oof_ensemble) * 100
    
    print("="*50)
    print(f"OOF R2 Score LightGBM: {score_lgb:.4f}")
    print(f"OOF R2 Score XGBoost:  {score_xgb:.4f}")
    print(f"OOF R2 Score CatBoost: {score_cb:.4f}")
    print(f"OOF R2 Score Ensemble: {score_ensemble:.4f}")
    print("="*50)
    
    print("\nStep 5: Generating Final Submission File...")
    final_preds = (test_lgb + test_xgb + test_cb) / 3.0
    
    submission = pd.DataFrame({
        'Index': test_feat['Index'].astype(int),
        'demand': np.clip(final_preds, 0.0, 1.0)
    })
    
    # Sort just in case to be perfectly safe
    submission = submission.sort_values('Index').reset_index(drop=True)
    
    test_orig = pd.read_csv('test.csv')
    assert (submission['Index'].values == test_orig['Index'].values).all(), "Index values do not match test file"
    
    submission.to_csv('submission.csv', index=False)
    print("Submission file successfully saved to submission.csv!")

if __name__ == "__main__":
    train_and_evaluate()
