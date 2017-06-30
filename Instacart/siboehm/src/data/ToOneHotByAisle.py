import pandas as pd
from loadData import DataLoader
from sklearn import preprocessing

data = DataLoader().load_data(files=[
    "order_products__train", "order_products__prior", "orders", "products"
])

# Contains which user ordered which order + more info about the order
orders = data["orders"]
# Contains the product for each orders
# Three orders for every user
order_prod_p = data["order_products__prior"]
# One order for every user, this order was made after the orders
# in order_prod_p (target)
order_prod_t = data["order_products__train"]
# Containing name, aisle and department of each product
product_info = data["products"]

# Map specific product_id to aisle_id of this product
order_prod_p_aisle = order_prod_p["product_id"].map(
    lambda x: product_info[product_info["product_id"] == x].iloc[0][2])

# Replace to product_id with the aisle of this product
order_prod_p = order_prod_p.drop("product_id", axis=1)
order_prod_p = order_prod_p.join(order_prod_p_aisle)
order_prod_p.rename(columns={'product_id': 'aisle_id'}, inplace=True)

# Remove some columns that are not needed for now
order_prod_p.drop(["reordered", "add_to_cart_order"], axis=1, inplace=True)

# Make a one hot encoding of the aisles
binarizer = preprocessing.LabelBinarizer()
binarized = binarizer.fit_transform(order_prod_p["aisle_id"])
bin_df = pd.DataFrame(binarized)
bin_df.columns = binarizer.classes_

order_prod_p.drop("aisle_id", axis=1, inplace=True)
order_prod_p = order_prod_p.join(bin_df)
order_prod_p = order_prod_p.groupby("order_id").sum()

master_p = pd.merge(order_prod_p, orders, right_on="order_id", left_index=True)
master_p.drop(
    [
        "order_id", "order_dow", "days_since_prior_order", "order_hour_of_day",
        "order_number"
    ],
    axis=1,
    inplace=True)
master_p_user = master_p.groupby("user_id").sum()
master_p_user.to_csv("ReduceToAisle.csv")
