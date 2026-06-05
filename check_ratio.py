import pandas as pd
train = pd.read_csv('train.csv')

# Parse time
train['hour'] = train['timestamp'].apply(lambda x: int(x.split(':')[0]))
train['minute'] = train['timestamp'].apply(lambda x: int(x.split(':')[1]))
train['time_minutes'] = train['hour'] * 60 + train['minute']

# Morning data
mornings = train[train['time_minutes'] <= 120]

day48_mean = mornings[mornings['day'] == 48]['demand'].mean()
day49_mean = mornings[mornings['day'] == 49]['demand'].mean()

print(f"Day 48 morning mean: {day48_mean:.5f}")
print(f"Day 49 morning mean: {day49_mean:.5f}")
print(f"Global scaling ratio: {day49_mean / day48_mean:.5f}")
