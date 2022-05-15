# Shopify Fall 2022 Data Science Intern Challenge
## Question 1:
On Shopify, we have exactly 100 sneaker shops, and each of these shops sells only one model of shoe. We want to do some analysis of the average order value (AOV). When we look at orders data over a 30 day window, we naively calculate an AOV of $3145.13. Given that we know these shops are selling sneakers, a relatively affordable item, something seems wrong with our analysis. 
a. Think about what could be going wrong with our calculation. Think about a better way to evaluate this data. 
b. What metric would you report for this dataset?
c. What is its value?

### Exploratory Analysis:
First, I want to take a glimpse at the given data and replicate the calculation of naive average order value.
```python
# import modules
from matplotlib import style
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
```python
# import dataset
shopify_df = pd.read_csv("shopify_dataset.csv")
print(shopify_df.head) # Look at the first 5 rows of dataset
```
```
Output:
  order_id  shop_id  user_id  order_amount  total_items payment_method           created_at
0         1       53      746           224            2           cash  2017-03-13 12:36:56
1         2       92      925            90            1           cash  2017-03-03 17:38:52
2         3       44      861           144            1           cash   2017-03-14 4:23:56
3         4       18      935           156            1    credit_card  2017-03-26 12:43:37
4         5       18      883           156            1    credit_card   2017-03-01 4:35:11
```
We can use the .mean() function on order_amount column. This is the naive calculation of average order value.
```python
naive_aov = shopify_df['order_amount'].mean()
print("Naive calculation of Average Order Value for the dataset: ${:.2f}".format(naive_aov))
```
```
Output:
Naive calculation of Average Order Value for the dataset: $3145.13
```
This number seems to be wrong with sneakers being affordable goods. We can check the distribution of order_amount to see if there are any outliers.
```python
sns.set(style="darkgrid")
g = sns.boxplot(x = shopify_df['order_amount'])
plt.show()
```
```
Output:
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

```
