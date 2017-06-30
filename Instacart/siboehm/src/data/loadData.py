import pandas as pd
import numpy as np
"""
Class to load data from disk into memory
"""


class DataLoader(object):

    # For the bigger files it makes sense to manually
    # set file type to reduce memory usage
    _types = {
        "products": {
            "aisle_id": np.int8,
            "department_id": np.int8,
            "product_id": np.int32
        },
        "aisles": {},
        "departments": {},
        "order_products__prior": {
            "order_id": np.int32,
            "add_to_cart_order": np.int16,
            "reordered": np.int8,
            "product_id": np.int32
        },
        "order_products__train": {
            "order_id": np.int32,
            "add_to_cart_order": np.int16,
            "reordered": np.int8,
            "product_id": np.int32
        },
        "orders": {
            "order_dow": np.int8,
            "order_hour_of_day": np.int8,
            "order_number": np.int16,
            "order_id": np.int32,
            "user_id": np.int32,
            "days_since_prior_order": np.float32
        }
    }

    def __init__(self, data_path="../../data/raw/"):
        self.path = data_path

    def load_data(self,
                  files=[
                      "aisles", "departments", "order_products__train",
                      "order_products__prior", "orders", "products"
                  ]):
        """
        Load the given files into memory,
        using smarter data types for the columns
        """
        data = {}
        for f in files:
            print("Loading", f, "...")
            data[f] = pd.read_csv(self.path + f + ".csv", dtype=self._types[f])
        return data

    def load_data_no_types(self,
                           files=[
                               "aisles", "departments", "order_products__train",
                               "order_products__prior", "orders", "products"
                           ]):
        """
        Load the given files into memory, without
        using smarter data types for the columns
        """
        data = {}
        for f in files:
            print("Loading", f, "...")
            data[f] = pd.read_csv(self.path + f + ".csv")
        return data
