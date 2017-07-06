import pandas as pd
import numpy as np


class DataLoader(object):
    def __init__(self, data_path="./data"):
        self.path = data_path

        self.dtypes = {
            "order_products__prior": {
                'order_id': np.int32,
                'product_id': np.uint16,
                'add_to_cart_order': np.int16,
                'reordered': np.int8},
            "order_products__train": {
                'order_id': np.int32,
                'product_id': np.uint16,
                'add_to_cart_order': np.int16,
                'reordered': np.int8},
            "orders": {
                'order_id': np.int32,
                'user_id': np.int32,
                'eval_set': 'category',
                'order_number': np.int16,
                'order_dow': np.int8,
                'order_hour_of_day': np.int8,
                'days_since_prior_order': np.float32},
            "products": {
                'product_id': np.uint16,
                'order_id': np.int32,
                'aisle_id': np.uint8,
                'department_id': np.uint8},
            "aisles": {
                'aisle_id': np.uint8,
                'aisle': str},
            "departments": {
                'department_id': np.uint8,
                'department': str}
        }

    def load_raw_files(self, files=["aisles", "departments", "order_products", "orders", "products"]):
        """
        Loads all given files with correct dtypes into a dict.
        Loads all files per default.

        :param files: list of files to load
        :return: dict with data of files
        """

        data = {}
        for f in files:
            data[f] = self.load_raw_file(f)

        return data

    def load_raw_file(self, file):
        """
        Loads given file with correct dtypes.

        file = "order_products" returns a single concatenated DataFrame of "order_products__prior" and "order_products__test"

        :param file: filename
        :return: DataFrame of file content
        """
        # Load both order_product files
        if file == "order_products":
            prior = self.load_raw_file("order_products__prior")
            train = self.load_raw_file("order_products__train")

            return pd.concat([prior, train])

        # Load file
        data = pd.read_csv(self.path + "/raw/" + file + ".csv", delimiter=',',
                           low_memory=True, engine='c', encoding="latin1",
                           dtype=self.dtypes[file])
        return data

    def load_master_file(self, usecols=None):
        # Define data types to save ram usage
        data_type = {"order_id": int, "user_id": int, "eval_set": str, "order_number": np.int16, "order_dow": np.int8,
                     "order_hour_of_day": np.int8, "days_since_prior_order": np.float16, "product_id": np.float16,
                     "add_to_cart_order": np.float16, "reordered": np.float16, "product_name": str, "aisle_id": np.float8,
                     "department_id": np.float8, "aisle": str, "department": str}
        master = pd.read_csv(self.path + "/interim/master.csv", delimiter=',',
                             low_memory=True, engine='c', encoding="latin1",
                             dtype=data_type, usecols=usecols)

        return master

    def create_master(self):
        data = self.load_raw_files()

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
