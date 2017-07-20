import pandas as pd
import datetime
from loadData import DataLoader
from sklearn import preprocessing
"""
Converts the data in the following way:
1. Map individual products to their aisle, thereby reducing dimension of
product-space from ~50.000 to 134
2. Flatten the data to the following structure:
---------------------------------------------------------------------------------------------
user-id | # product orders of user from aisle 1 | # product orders of user from aisle 2 | ... | aisle 134
user-id | # product orders of user from aisle 1 | # product orders of user from aisle 2 | ... | aisle 134
.
.
---------------------------------------------------------------------------------------------
"""

data = DataLoader().load_data(files=[
    "raw/order_products__train", "raw/order_products__prior", "raw/orders",
    "raw/products"
])

# Contains which user ordered which order + more info about the order
orders = data["raw/orders"]
# Contains the product for each orders
# Three orders for every user
order_prod_p = data["raw/order_products__prior"]
# One order for every user, this order was made after the orders
# in order_prod_p (target)
order_prod_t = data["raw/order_products__train"]
# Containing name, aisle and department of each product
product_info = data["raw/products"]


# Warning, function is not side effect free
def reduceToAisle(df):
    # Map product to the aisle the product is located in
    df_aisle = df["product_id"].map(
        lambda x: product_info[product_info["product_id"] == x].iloc[0][2])

    # Replace the product id with the aisle of this product
    df = df.drop("product_id", axis=1)
    df = df.join(df_aisle)
    df.rename(columns={'product_id': 'aisle_id'}, inplace=True)

    # Remove some columns not needed for now
    df.drop(["reordered", "add_to_cart_order"], axis=1, inplace=True)

    # Make a one hot encoding of the aisles
    binarizer = preprocessing.LabelBinarizer()
    binarized = binarizer.fit_transform(df["aisle_id"])
    bin_df = pd.DataFrame(binarized)
    classes_binarized = binarizer.classes_

    # Fix overflow of classes :/
    for i in range(len(classes_binarized)):
        if classes_binarized[i] < 0:
            classes_binarized[i] = classes_binarized[i] + 128 + 128

    bin_df.columns = classes_binarized

    df.drop("aisle_id", axis=1, inplace=True)
    df = df.join(bin_df)
    df = df.groupby("order_id").sum()

    master = pd.merge(df, orders, right_on="order_id", left_index=True)
    master.drop(
        [
            "order_id", "order_dow", "days_since_prior_order",
            "order_hour_of_day", "order_number"
        ],
        axis=1,
        inplace=True)
    master_user = master.groupby("user_id").sum()
    return master_user


# Transforming 1.000.000 individual products order takes about 10 minutes on my system

print("Transforming prior...")
start_time = datetime.datetime.now()
order_prod_p = reduceToAisle(order_prod_p)
duration = datetime.datetime.now() - start_time
print("Transformation finished after {} hours, {} minutes, {} seconds".format(
    duration.seconds // 3600, duration.seconds // 60 % 60, duration.seconds %
    60))
order_prod_p.to_csv("../../data/processed/ReducedToAislePrior.csv")

print("Transforming train...")
start_time = datetime.datetime.now()
order_prod_t = reduceToAisle(order_prod_t)
duration = datetime.datetime.now() - start_time
print("Transformation finished after {} hours, {} minutes, {} seconds".format(
    duration.seconds // 3600, duration.seconds // 60 % 60, duration.seconds %
    60))

order_prod_p.to_csv("../../data/processed/ReducedToAisleTrain.csv")
