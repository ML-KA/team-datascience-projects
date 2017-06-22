import pandas as pd
import numpy as np


class Preprocessor(object):
    def __init__(self, data_path="./data"):
        self.path = data_path

    def load_data_raw(self, files=["aisles", "departments", "order_products", "orders", "products"]):
        # Load both order_product files
        order_products = False
        if "order_products" in files:
            order_products = True

            files.remove("order_products")
            files.append(["order_products__train", "order_products__prior"])

        # Load files
        data = {}
        for f in files:
            data[f] = pd.read_csv(self.path + "/raw/" + f + ".csv", delimiter=',')

        # Join both order_products files
        if order_products:
            data["order_products"] = pd.concat([data["order_products__prior"], data["order_products__train"]])
            del data["order_products__prior"], data["order_products__train"]

        return data

    def load_data_master(self, usecols=None):
        # Define data types to save ram usage
        data_type = {"order_id": int, "user_id": int, "eval_set": str, "order_number": np.int16, "order_dow": np.int8,
                 "order_hour_of_day": np.int8, "days_since_prior_order": np.float16, "product_id": np.float32,
                 "add_to_cart_order": np.float16, "reordered": np.float16, "product_name": str, "aisle_id": np.float32,
                 "department_id": np.float32, "aisle": str, "department": str}
        master = pd.read_csv(self.path + "/interim/master.csv", delimiter=',',
                             low_memory=True, engine='c', encoding="latin1",
                             dtype=data_type, usecols=usecols)

        return master

    def create_master(self, data):
        data = self.load_data_raw()

        # Join everything over orders
        master = pd.merge(data["orders"], data["order_products"], on="order_id", how="left")
        del data["orders"], data["order_products"]

        master = pd.merge(master, data["products"], on="product_id", how="left")
        del data["products"]

        master = pd.merge(master, data["aisles"], on="aisle_id", how="left")
        del data["aisles"]

        master = pd.merge(master, data["departments"], on="department_id", how="left")
        del data["departments"], data

        master.to_csv(self.path + "/interim/master.csv", encoding="latin1", index=False)
