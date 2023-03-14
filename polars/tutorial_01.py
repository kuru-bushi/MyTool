#%%

import numpy as np # linear algebra
import polars as pl # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# %%
train_data = pl.read_csv("./data/titanic_train.csv")
train_data.head()
# %%
women = train_data.filter(pl.col('Sex')=='female')['Survived']
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)
# %%
men = train_data.filter(pl.col('Sex')=='male')['Survived']
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
# %%
from sklearn.ensemble import RandomForestClassifier

test_data = pl.read_csv("./data/titanic_test.csv")
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pl.get_dummies(train_data.select(features), columns='Sex')
X_test = pl.get_dummies(test_data.select(features), columns='Sex')

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X.to_numpy(), y.to_numpy())
predictions = model.predict(X_test.to_numpy())

#%%
import time
def print_accuracy(f):
    print("Root mean squared test error = {0}".format(np.sqrt(np.mean((f(X_test) - y_test)**2))))
    time.sleep(0.5) # to let the print get out before any progress bars

#%%


# %%
output = pl.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.write_csv('submission.csv')
print("Your submission was successfully saved!")
# %%
train_data.glimpse()
# %%
train_data.describe()
# %%
train_data.get_columns()
#%%
train_data.head()
# %%
train_data.select([pl.col("Age")*4])
# %%
train_data.filter(pl.col("Age")> 10)
# %%
train_data.select("Age")
# %% 追加
train_data["Age_10"] = train_data.with_columns(pl.col("Age_10")*10)
train_data.head()

# %% 更新
train_data.with_columns(pl.col("Age")*10).head()

# %%
