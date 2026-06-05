import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('train.csv')
df = df.dropna(subset=['demand'])

df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
df['minute'] = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
df['time_minutes'] = df['hour'] * 60 + df['minute']

X = df[['NumberofLanes', 'Temperature', 'time_minutes']].copy()
X = X.fillna(0)
y = df['demand']

lr = LinearRegression()
lr.fit(X, y)
print(f"Linear Regression R2: {lr.score(X, y)}")

# Try a single Decision Tree to see if it can easily find a perfect step function
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(max_depth=5)
dt.fit(X, y)
print(f"Decision Tree (depth 5) R2: {dt.score(X, y)}")
