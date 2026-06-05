import pandas as pd
from src.utils import decode_geohash

def build_features(df, train_df=None):
    """
    Build basic tabular features for demand prediction.
    No historical lag features, just raw deterministic columns.
    """
    print("Building raw tabular features...")
    df = df.copy()
    
    # 1. Fill categorical NAs
    for col in ['RoadType', 'Weather', 'LargeVehicles', 'Landmarks']:
        df[col] = df[col].fillna('Unknown')
    
    # 2. Extract Time
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_minutes'] = df['hour'] * 60 + df['minute']
    
    # 3. Impute Temperature
    # Median temperature of the same day and hour
    df['Temperature'] = df.groupby(['day', 'hour'])['Temperature'].transform(lambda x: x.fillna(x.median()))
    # Fallbacks
    df['Temperature'] = df.groupby('day')['Temperature'].transform(lambda x: x.fillna(x.median()))
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    
    # 4. Geo features
    df['geohash_prefix4'] = df['geohash'].str[:4]
    df['geohash_prefix5'] = df['geohash'].str[:5]
    
    lat_lon = df['geohash'].apply(lambda x: pd.Series(decode_geohash(x)))
    df['latitude'] = lat_lon[0]
    df['longitude'] = lat_lon[1]
    
    return df
