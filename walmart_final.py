

import pandas as pd
get_ipython().magic(u'matplotlib inline')

get_ipython().system(u'pwd')

df = pd.read_csv('train.csv')

df['Date'] = pd.to_datetime(df['Date'])

df.head()

df['Store'].value_counts(normalize=True).plot(kind = 'barh')

df.plot(x="Date", y = "IsHoliday")
# One-hot-encode "Store" and "Dept"
df = pd.get_dummies(df, columns=['Store', 'Dept'])

df.head()
# Extract date features
df['Date_dayofweek'] = df['Date'].dt.dayofweek
df['Date_month'] = df['Date'].dt.month
df['Date_year'] = df['Date'].dt.year
df['Date_day'] = df['Date'].dt.day

# Extract time-lag features for 1 day, 2 day, 3 day, 5 day, 1 week, 2 week, and a month ago
for days_to_lag in [1, 2, 3, 5, 7, 14, 30]:
    df['Weekly_sales_lag_{}'.format(days_to_lag)] = df.Weekly_Sales.shift(days_to_lag)

df.head()

# Replace all NaN values with 0
df = df.fillna(0)

df.IsHoliday = df.IsHoliday.astype(int)

# Grab features and target
# Remove date from features because it's overly-unique
# Remove weekly_sales from features since it's the target and
# we don't have access to it at the time of prediction
x = df[df.columns.difference(['Date', 'Weekly_Sales'])]  
y = df.Weekly_Sales

x.head()

y[:3]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3)


x_train.shape

x_test.shape


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


clf = LinearRegression()
clf.fit(x_train, y_train)

# Calculate R2
clf.score(x_test, y_test)

get_ipython().run_cell_magic(u'time', u'', u'clf = RandomForestRegressor(n_jobs=-1)  # use all cores\nclf.fit(x_train, y_train)')

# Better R2 with random forest
# You can probably do hyper-parameter grid/random search to improve
clf.score(x_test, y_test)

# Other regression metrics
# http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Mean absolute error
predicted = clf.predict(x_test)
mean_absolute_error(y_test, predicted)

# MSE
mean_squared_error(y_test, predicted)

