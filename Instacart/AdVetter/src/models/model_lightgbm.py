import datetime
import lightgbm as lgb
import pandas as pd
import numpy as np
import gc

from data.data_loader import DataLoader
from data.preprocessor import Preprocessor

# Load files
data_loader = DataLoader()
order_products = data_loader.load_raw_file('order_products')
orders = data_loader.load_raw_file('orders')
products = data_loader.load_raw_file('products')

data = orders.merge(order_products, on='order_id', how='left')

prior = data[data.eval_set == 'prior']
train = data[data.eval_set == 'train']
test = data[data.eval_set == 'test']

# Remove products that were first ordered in the user's last order (we don't have to predict those)
train = train[train.reordered != 0]

# Column not given for test data
train.drop('add_to_cart_order', inplace=True, axis=1)

# All reorders per user
reorders = train.groupby('user_id').product_id.apply(set)

# All products user has bought
all_products = prior.groupby('user_id').product_id.apply(set)

not_reordered = all_products - reorders

del data, train, test


# Create test and train set
preprocessor = Preprocessor()
train_df = preprocessor.create_train_set(all_products, orders, products, reorders, not_reordered)
test_df = preprocessor.create_test_set(all_products, orders, products)


############################################
# Create Features
############################################
# Product Features
prods = pd.DataFrame()

prods['product_orders'] = prior.groupby(prior.product_id).size().astype(np.int32)
prods['product_reorders'] = prior['reordered'].groupby(prior.product_id).sum().astype(np.float32)
prods['product_reorder_rate'] = (prods.product_reorders / prods.product_orders).astype(np.float32)

train_df = train_df.join(prods, on='product_id', how='left')
test_df = test_df.join(prods, on='product_id', how='left')

del prods
gc.collect()

# User Features
users = pd.DataFrame()

users['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)

users['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)
users['total_items'] = prior.groupby('user_id').size().astype(np.int16)
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)

users['all_products'] = prior.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

# number of reordered items per user
users['u_reordered'] = prior.groupby('user_id')['reordered'].sum().astype(np.float32)

users['u_reorder_rate'] = users['u_reordered']/users['total_items']

users.drop('all_products', axis=1, inplace=True)

train_df = train_df.join(users, on='user_id', how='left')
test_df = test_df.join(users, on='user_id', how='left')

del users
gc.collect()

# User / Product features
users_prods = pd.DataFrame()

# count how often a user bought each product
users_prods['up_count'] = prior.groupby(['user_id', 'product_id']).size().astype(np.int16)

train_df = train_df.set_index(['user_id', 'product_id'])
train_df = train_df.join(users_prods)

test_df = test_df.set_index(['user_id', 'product_id'])
test_df = test_df.join(users_prods)

del users_prods
gc.collect()

# Other features

prior = prior.merge(products, on='product_id', how='left')

# reorder_rate aisle
aisles = pd.DataFrame()
aisles['a_reordered'] = prior.groupby('aisle_id')['reordered'].sum().astype(np.float32)
aisles['a_total_items'] = prior.groupby('aisle_id').size().astype(np.float32)
aisles['a_reorder_rate'] = aisles['a_reordered']/aisles['a_total_items']

train_df.reset_index(inplace=True)
train_df = train_df.join(aisles, on='aisle_id', how='left')

test_df.reset_index(inplace=True)
test_df = test_df.join(aisles, on='aisle_id', how='left')

# reorder_rate department
departments = pd.DataFrame()
departments['d_reordered'] = prior.groupby('department_id')['reordered'].sum().astype(np.float32)
departments['d_total_items'] = prior.groupby('department_id').size().astype(np.float32)
departments['d_reorder_rate'] = departments['d_reordered']/departments['d_total_items']

train_df = train_df.join(departments, on='department_id', how='left')
test_df = test_df.join(departments, on='department_id', how='left')

del aisles, departments
gc.collect()


# Weekday
train_df['weekday_sin'] = np.sin(2 * np.pi * train_df['order_dow'] / 7)
train_df['weekday_cos'] = np.cos(2 * np.pi * train_df['order_dow'] / 7)

test_df['weekday_sin'] = np.sin(2 * np.pi * test_df['order_dow'] / 7)
test_df['weekday_cos'] = np.cos(2 * np.pi * test_df['order_dow'] / 7)

# Hour

train_df['hour_sin'] = np.sin(2 * np.pi * train_df['order_hour_of_day'] / 24)
train_df['hour_cos'] = np.cos(2 * np.pi * train_df['order_hour_of_day'] / 24)

test_df['hour_sin'] = np.sin(2 * np.pi * test_df['order_hour_of_day'] / 24)
test_df['hour_cos'] = np.cos(2 * np.pi * test_df['order_hour_of_day'] / 24)

train_df['days_since_ratio'] = train_df.days_since_prior_order / train_df.average_days_between_orders
test_df['days_since_ratio'] = test_df.days_since_prior_order / test_df.average_days_between_orders

train_df['up_orders_ratio'] = (train_df['up_count'] / train_df['nb_orders']).astype(np.float32)
test_df['up_orders_ratio'] = (test_df['up_count'] / test_df['nb_orders']).astype(np.float32)

############################################
# Train Model
############################################

X_test = test_df

X_train = train_df.ix[:, train_df.columns != 'reordered']
y_train = train_df['reordered']

del test_df, train_df, products, order_products

X_train['product_id'] = X_train['product_id'].astype(int)
X_test['product_id'] = X_test['product_id'].astype(int)

lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=['aisle_id', 'department_id', 'product_id'])


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 96,
    'max_depth': 10,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5
}
rounds = 100

clf = lgb.train(params, lgb_train, rounds)

predictions = clf.predict(X_test)

#######################################
# Save Result
#######################################
X_test['reordered'] = predictions

THRESHOLD = 0.22  # guess, should be tuned with crossval on a subset of train data

results = dict()
for row in X_test.itertuples():
    if row.reordered > THRESHOLD:
        try:
            results[row.order_id] += ' ' + str(row.product_id)
        except:
            results[row.order_id] = str(row.product_id)

test_orders = orders[orders.eval_set == "test"]
for order in test_orders.order_id:
    if order not in results:
        results[order] = 'None'

sub = pd.DataFrame.from_dict(results, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S_submission.csv")
sub.to_csv(filename, index=False)
