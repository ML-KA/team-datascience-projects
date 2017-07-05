import lightgbm as lgb
import pandas as pd

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

del data, prior, train


# Create test and train set
preprocessor = Preprocessor()
train_df = preprocessor.create_train_set(all_products, orders, products, reorders, not_reordered)
test_df = preprocessor.create_test_set(all_products, orders, products)


############################################
# Train Model
############################################

X_test = test_df

X_train = train_df.ix[:, train_df.columns != 'reordered']
y_train = train_df['reordered']

del test_df, train_df, products, orders, order_products

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
#######################################
#######################################
X_test['reordered'] = predictions

THRESHOLD = 0.22  # guess, should be tuned with crossval on a subset of train data

d = dict()
for row in X_test.itertuples():
    if row.reordered > THRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except:
            d[row.order_id] = str(row.product_id)

test_orders = orders[orders.eval_set == "test"]
for order in test_orders.order_id:
    if order not in d:
        d[order] = 'None'

sub = pd.DataFrame.from_dict(d, orient='index')

sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('sub.csv', index=False)
