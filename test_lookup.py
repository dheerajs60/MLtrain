import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_day48 = train[train['day'] == 48][['geohash', 'timestamp', 'demand']]
train_day48 = train_day48.rename(columns={'demand': 'demand_lookup'})

test_lookup = test.merge(train_day48, on=['geohash', 'timestamp'], how='left')

missing = test_lookup['demand_lookup'].isna().sum()
total = len(test)

print(f"Total test rows: {total}")
print(f"Missing in Day 48: {missing}")
print(f"Match percentage: {100 - (missing / total * 100):.2f}%")

