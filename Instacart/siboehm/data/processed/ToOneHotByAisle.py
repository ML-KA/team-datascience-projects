import pandas as pd

# Contains which user ordered which order + more info about the order
orders = pd.read_csv("../raw/orders.csv")
# Contains the product for each orders
# Three orders for every user
order_prod_p = pd.read_csv("../raw/order_products__prior.csv").head(100)
# One order for every user, this order was made after the orders
# in order_prod_p (target)
order_prod_t = pd.read_csv("../raw/order_products__train.csv")
# Containing name, aisle and department of each product
product_info = pd.read_csv("../raw/products.csv")

# Map specific product_id to aisle_id of this product
order_prod_p_aisle = order_prod_p["product_id"].map(
    lambda x: product_info[product_info["product_id"] == x].iloc[0][2])

# Replace to product_id with the aisle of this product
order_prod_p = order_prod_p.drop("product_id", axis=1)
order_prod_p = order_prod_p.join(order_prod_p_aisle)
order_prod_p.rename(columns={'product_id': 'aisle_id'}, inplace=True)

# Remove some columns that are not needed for now
order_prod_p.drop(["reordered", "add_to_cart_order"], axis=1, inplace=True)
