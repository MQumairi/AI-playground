# %%
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%

# Load boston houses dataset
boston_dataset = load_boston()
print(boston_dataset["DESCR"])
# Convert from Bunch to DataFrame
data = pd.DataFrame(data=boston_dataset.data,
                    columns=boston_dataset.feature_names)

# Adding the target
data["PRICE"] = boston_dataset.target

data.head()
# %%

# Building a Histogram with plt
plt.hist(data["PRICE"], bins=50)
plt.show()

# Builidng a Histogram with sns
sns.distplot(data["PRICE"])
# %%

# Means
# print(data.mean())
# print(boston_dataset.DESCR)

frequency = data["RAD"].value_counts()

plt.bar(frequency.index, frequency)
# %%

# The correlation between the Price and the room size:
data.corr()

# %%
mask = np.zeros_like(data.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True
mask
# %%
plt.figure(figsize=(16, 10))
sns.heatmap(data.corr(), cmap=cm.coolwarm, mask=mask,
            annot=True, annot_kws={"size": 14})
sns.set_style("white")
plt.show()

# %%
print(boston_dataset["DESCR"])
# %%
nox_dis_correlation = data["NOX"].corr(data["DIS"])
plt.figure(figsize=(16, 10))
plt.scatter(data["DIS"], data["NOX"], alpha=0.6, s=100, color="indigo")
plt.title("DIS vs NOX, correlation = " +
          str(round(nox_dis_correlation, 3)), fontsize=16)
plt.xlabel("Distance from Employment Center", fontsize=14)
plt.ylabel("Nitric Oxide Polution", fontsize=14)
plt.show()

# %%
sns.set()
sns.set_style("whitegrid")
sns.jointplot(x=data["DIS"], y=data["NOX"], size=7,
              color="indigo", joint_kws={"alpha": 0.6})
# %%

# %%

sns.lmplot(x="RM", y="PRICE", data=data, size=7)
# %%
# %%time
# sns.pairplot(data, kind="reg", plot_kws={"line_kws": {"color": "cyan"}})
# plt.show()
# %%
# Train and split data set split
prices = data["PRICE"]
features = data.drop('PRICE', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    features, prices, test_size=0.2, random_state=10)

linearRegression = LinearRegression()
linearRegression.fit(X_train, y_train)
print("Intercept is: " + str(linearRegression.intercept_))
pd.DataFrame(data=linearRegression.coef_, index=X_train.columns)

print(linearRegression.score(X_train, y_train))
print(linearRegression.score(X_test, y_test))

# %%
