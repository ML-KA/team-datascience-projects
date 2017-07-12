import pandas as pd
import datetime
from loadData import DataLoader
"""
Clean the train dataset
We only need to predict products that have been ordered by this
user previously, the rest can be removed
"""

data = DataLoader().load_data(files=["raw/order_products__train"])
train = data["raw/order_products__train"]
print("Orders in train (raw):", train.shape[0])

# We only need the products that have been previously ordered
train = train[train.reordered == 1]
print("Orders in train (cleaned):", train.shape[0])

train.to_csv("../../data/processed/order_products__train_clean.csv")
