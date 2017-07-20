import pandas as pd
import numpy as np
import datetime
import gc
from loadData import DataLoader
from sklearn import preprocessing
"""
Converts the data in the following way:
1. Focus only on the 200 most reordered products, remove the rest thereby reducing dimension of
product-space from ~50.000 to 200
2. Flatten the data to the following structure:
---------------------------------------------------------------------------------------------
user-id 1 | # product orders product 1 | # product orders product 2 |...|# product orders product 200 |
user-id 2 | # product orders product 1 | # product orders product 2 |...|# product orders product 200 |
.
.
---------------------------------------------------------------------------------------------
"""

data = DataLoader().load_data(files=[
    "processed/order_products__train_clean", "raw/order_products__prior",
    "raw/orders"
])

# Contains which user ordered which order + more info about the order
orders = data["raw/orders"]
# Contains the product for each orders
# Three orders for every user
order_prod_p = data["raw/order_products__prior"].head(10000000)
# One order for every user, this order was made after the orders
# in order_prod_p (target)
# The data has been cleaned to contain only reorders, as this is our target
# order_prod_t = data["processed/order_products__train_clean"]

# The products that were reordered the most
print("Finding most reordered...")
most_reordered_prod = order_prod_p[order_prod_p.reordered == 1][
    "product_id"].value_counts().head(200).index

# Remove all order products that are not among the most reordered
print("Removing the rest...")
order_prod_p = order_prod_p[order_prod_p.product_id.isin(most_reordered_prod)]
order_prod_p.drop(["reordered", "add_to_cart_order"], axis=1)

# Multiclass to One hot
print("Generating one hot...")
binarizer = preprocessing.LabelBinarizer()
bin_df = pd.DataFrame(binarizer.fit_transform(order_prod_p["product_id"]))
bin_df.columns = binarizer.classes_
prod_dtypes = {}
for x in most_reordered_prod:
    prod_dtypes[x] = np.int8
order_prod_p.drop("product_id", axis=1, inplace=True)
order_prod_p = order_prod_p.join(bin_df)
order_prod_p = order_prod_p.groupby("order_id").sum()
del bin_df
gc.collect()

# Add more information about the orders
print("Joining with the orders...")
order_prod_p.drop(["add_to_cart_order", "reordered"], axis=1, inplace=True)
user_orders = pd.merge(
    order_prod_p, orders, right_on="order_id", left_index=True)
del order_prod_p
user_orders.drop(
    [
        "order_id", "order_dow", "days_since_prior_order", "order_hour_of_day",
        "order_number"
    ],
    axis=1,
    inplace=True)
gc.collect()

# Sum up get the amount of products ordered by user
print("Summing up...")
user_orders = user_orders.groupby("user_id").sum()

# Save the file
print("Saving...")
# user_orders.to_csv("../../data/processed/ReducedToMostOrderedFlat.csv")
