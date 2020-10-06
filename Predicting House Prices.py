# %%
from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%

# Load boston houses dataset
boston_dataset = load_boston()

# Convert from Bunch to DataFrame
data = pd.DataFrame(data=boston_dataset.data,
                    columns=boston_dataset.feature_names)

# Adding the target
data["PRICE"] = boston_dataset.target
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
