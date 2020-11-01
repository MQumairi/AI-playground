# %%
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
print(pd.DataFrame(data=linearRegression.coef_, index=X_train.columns))

print(linearRegression.score(X_train, y_train))
print(linearRegression.score(X_test, y_test))

# %%
data['PRICE'].skew()

y_log = np.log(data['PRICE'])

size_of_prices = np.arange(0, data['PRICE'].size)

# plt.scatter(size_of_prices, np_arr_prices)

# plt.scatter(size_of_prices, y_log, color="red")

sns.distplot(y_log)
plt.show()

# %%

# Rerunning regression using transformed data

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    features, y_log, test_size=0.2, random_state=10)

linearRegression_2 = LinearRegression()
linearRegression_2.fit(X_train_2, y_train_2)

print(linearRegression_2.score(X_train_2, y_train_2))
print(linearRegression_2.score(X_test_2, y_test_2))


print(np.pi)
# %%
x_incl_constant = sm.add_constant(X_train)

sm_model = sm.OLS(y_train_2, x_incl_constant)

results = sm_model.fit()

results.pvalues

print("mean is ", results.mse_resid)

pd.DataFrame({'coef': results.params, 'p-values': round(results.pvalues, 3)})
# %%

vif = []
for i in range(len(x_incl_constant.columns)):
    vif.append(variance_inflation_factor(x_incl_constant.values, i))


pd.DataFrame({"col": x_incl_constant.columns, "Vif": np.around(vif, 2)})
# variance_inflation_factor(x_incl_constant.values, 1)
# variance_inflation_factor(np.array(x_incl_constant), 2)

# print(len(x_incl_constant.columns)
# %%
