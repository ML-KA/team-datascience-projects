import pandas as pd

# Load files
files = ["order_products__prior", "order_products__train",
         "orders"]
data = {}
for f in files:
    data[f] = pd.read_csv("./data/raw/" + f + ".csv", delimiter=',')

data["order_products"] = pd.concat([data["order_products__prior"], data["order_products__train"]])
del data["order_products__prior"], data["order_products__train"]

# Get orders and user_ids to predict (test)
orders = data["orders"]
test = orders[orders["eval_set"] == "test"]
test_uids = test["user_id"]

orders_prior = orders[(orders["eval_set"] == "prior") & (orders["user_id"].isin(test_uids))]

# Get products of prior orders
products = data["order_products"]
products_prior = products[products["order_id"].isin(orders_prior["order_id"])]

# Get order_id of last order per user
orders_prior_ids = orders_prior.groupby("user_id")["order_number"].idxmax()
last_order_ids = orders_prior.loc[orders_prior_ids]["order_id"]

# Aggregate all products of same order to a list and select last orders
products_prior_list = pd.DataFrame(products_prior.groupby('order_id')['product_id'].apply(list))
products_last_order = products_prior_list.loc[last_order_ids]

# Merge to get user_id and list of product_ids
temp = pd.merge(products_last_order, orders_prior, left_index=True, right_on="order_id")
temp = temp[["user_id", "product_id"]]
temp.columns = ["user_id", "products"]

# Merge to connect user_id to order_id which has to be predicted
final = pd.merge(test, temp, on="user_id")
final = final[["order_id", "products"]]

# Write submission
final["products"] = final["products"].transform(lambda x: " ".join(str(y) for y in x))
final.to_csv("submission.csv", index=False)
