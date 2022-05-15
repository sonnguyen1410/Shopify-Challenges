# import modules
from matplotlib import style
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# importing dataset:
shopify_df = pd.read_csv("shopify_dataset.csv")
print(shopify_df.head())

# this is the naive calculation of average order value 
naive_aov = shopify_df['order_amount'].mean()
print("Naive calculation of Average Order Value for the dataset: ${:.2f}".format(naive_aov)) # it yields a value of 3,145.13, which seems wrong to our analysis of sneakers sales

### 1st theory: 
# my theory is that there are outliers that skew the average value, we can test this by using scatter plot to experiment visually
sns.set(style="darkgrid")
g = sns.boxplot(x = shopify_df['order_amount'])
plt.show() # We can see that there are some thousand-dollar orders that result in a greater value than we expected. 

# A better way to evaluate this data is to use median value to calculate Median Order Value
mov = shopify_df['order_amount'].median()
print("Median Order Value: ${:.2f}".format(mov)) # this yields a value of $284.00

### 2nd theory:
# My second theory is that there are some stores that sell high-end shoes or sell in bulk, which results 
# Stores that have outlier value orders (> $10,000)
high_store_set = set()
for index, row in shopify_df.iterrows():
    if int(row['order_amount']) > 10000:
        high_store_set.add(row['shop_id'])
print("Stores that have outlier value orders: {}".format(high_store_set)) # There are two stores that have outlier value orders. We can exclude them from the calculation.

# calculate Average Order Value excluding stores with these two stores
trimmed_df = shopify_df.loc[~shopify_df['shop_id'].isin(high_store_set)]
trimmed_aov = trimmed_df['order_amount'].mean()
print("Average Order Value after excluding outlier stores: ${:.2f}".format(trimmed_aov)) # this yields a value of $300.16, which is much more logical for sneakers.
