# import modules
from matplotlib import style
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# importing dataset:
shopify_df = pd.read_csv("shopify_dataset.csv")
print(shopify_df.head()) # Look at the first 5 rows of dataset

# this is the naive calculation of average order value 
naive_aov = shopify_df['order_amount'].mean()
print("Naive calculation of Average Order Value for the dataset: ${:.2f}".format(naive_aov)) # it yields a value of 3,145.13, which seems wrong to our analysis of sneakers sales

# Check the distribution of order_amount in a boxplot
sns.set(style="darkgrid")
g = sns.boxplot(x = shopify_df['order_amount'])
plt.show() # We can see that there are some thousand-dollar orders that result in a greater value than we expected. 

# summary statistics of order_amount:
print(shopify_df['order_amount'].describe())

# Stores that have orders > $1,000
high_store_set = set()
for index, row in shopify_df.iterrows():
    if int(row['order_amount']) > 1000:
        high_store_set.add(row['shop_id'])
print("Stores that have orders > $1,000: {}".format(high_store_set))

# check each store data to see any discrepancies
for s in high_store_set:
    print('Store {}'.format(s))
    print(shopify_df[shopify_df['shop_id'] == s])
    print('======================================')

# create a new dataframe to make changes
shopify_new_df = shopify_df.copy()

# replace inaccurate data in store 42
shopify_new_df['order_amount'] = shopify_new_df['order_amount'].replace([704000], 704)
shopify_new_df['total_items'] = shopify_new_df['total_items'].replace([2000], 2)

# replace inaccurate data in store 78
shopify_new_df['order_amount'] = np.where(shopify_new_df['shop_id']==78, shopify_new_df['order_amount']/100, shopify_new_df['order_amount'])

# check each store again
print('Store 42'.format())
print(shopify_new_df[shopify_new_df['shop_id'] == 42])
print('======================================')
print('Store 78'.format())
print(shopify_new_df[shopify_new_df['shop_id'] == 78])
print('======================================')

# Now we calculate the Average Order Value based on newly modified dataset
new_aov = shopify_new_df['order_amount'].mean()
print('The new Average Order Value is ${:.2f}'.format(new_aov)) # this value makes more sense to our data

# Check the new distribution of order_amount in a boxplot
sns.set(style="darkgrid")
g = sns.boxplot(x = shopify_new_df['order_amount'])
plt.show() # The data are more condensed
