"""
This is a baseline prediction for evaluation of further models

It will just suggest all the products that were ordered in the last
order of each user
"""

import pandas as pd

orders = pd.read_csv("../../data/raw/orders.csv")

# Get users whose next reorders we want to predict
test_orders = orders[orders["eval_set"] == "test"]
test_uids = test_orders["user_id"]

# Prior orders of users whose next reorders we want to predict
orders_prior = orders[(orders["eval_set"] == "prior") &
                      (orders["user_id"].isin(test_uids))]

# The order_ids of the last order for each user to predict
last_orders_indexes = orders_prior.groupby("user_id")["order_number"].idxmax()
last_order_ids = orders_prior.loc[last_orders_indexes]["order_id"]

# Products of the prior orders of users that we target for our prediction
order_products = pd.read_csv("../../data/raw/order_products__prior.csv")
last_order_products = order_products[order_products["order_id"].isin(
    last_order_ids)]

# The products of the last orders for the targeted users
last_order_products_flat = pd.DataFrame(
    last_order_products.groupby("order_id")["product_id"].apply(list))

# In the table of order numbers and products for the last orders
# replace the order number with the user_id who ordered the products
last_products_by_user = pd.merge(
    last_order_products_flat,
    orders_prior,
    left_index=True,
    right_on="order_id")[["user_id", "product_id"]]
last_products_by_user.columns = ["user_id", "products"]

# Predict that the next order for each user will be exactly his last order
prediction = pd.merge(
    test_orders, last_products_by_user, on="user_id")[["order_id", "products"]]

# Write the submission in the correct format
prediction["products"] = prediction["products"].apply(
    lambda x: " ".join(str(y) for y in x))
prediction.to_csv("../submissions/baseline.csv", index=False)
