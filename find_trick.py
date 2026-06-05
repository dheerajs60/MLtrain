import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Find a matching row in Day 48 for test row 0
print("Test Row 0:")
print(test.iloc[0])

match_day48 = train[(train['day'] == 48) & (train['geohash'] == test.iloc[0]['geohash']) & (train['timestamp'] == test.iloc[0]['timestamp'])]
print("\nMatch in Day 48 Train:")
if len(match_day48) > 0:
    print(match_day48.iloc[0])
    print(f"\nDemand for Day 48: {match_day48.iloc[0]['demand']}")
else:
    print("No match found in Day 48")

# Let's check if the features in Day 49 test are exactly identical to Day 48 train
test_day49 = test.sort_values(['geohash', 'timestamp']).reset_index(drop=True)
train_day48 = train[(train['day'] == 48) & (train['timestamp'].isin(test['timestamp'].unique()))]
train_day48 = train_day48.sort_values(['geohash', 'timestamp']).reset_index(drop=True)

# Are they identically ordered and sized?
print(f"\nSize of test: {len(test_day49)}")
print(f"Size of train Day 48 (matching times): {len(train_day48)}")

