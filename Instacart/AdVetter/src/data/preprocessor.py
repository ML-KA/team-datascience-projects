from tqdm import tqdm

import pandas as pd


class Preprocessor(object):
    def __init__(self):
        pass

    @staticmethod
    def create_train_set(all_products, orders, products, reorders, not_reordered):

        order_list = []
        product_list = []
        labels = []

        # Loop over all train_orders to get a list with all reorders taken on last order with reordered = 1
        # and all other products the user has bought with reordered = 0
        train_orders = orders[orders.eval_set == 'train']

        for row in tqdm(train_orders.itertuples(), total=len(train_orders), desc="Building Trainset"):

            # try/catch faster than 'if row.user_id in reorders.index'
            try:
                # Add all reorders of user and corresponding order_id to list
                product_list += reorders[row.user_id]
                order_list += [row.order_id] * len(reorders[row.user_id])
                labels += [1] * len(reorders[row.user_id])

                # Add all not reordered products of user and corresponding order_id to list
                product_list += not_reordered[row.user_id]
                order_list += [row.order_id] * len(not_reordered[row.user_id])
                labels += [0] * len(not_reordered[row.user_id])

            except:
                # No reorders, so add all products of user and corresponding order_id to list
                product_list += all_products[row.user_id]
                order_list += [row.order_id] * len(all_products[row.user_id])
                labels += [0] * len(all_products[row.user_id])

        train_df = pd.DataFrame()
        train_df['order_id'] = order_list
        train_df['product_id'] = product_list
        train_df['reordered'] = labels
        del order_list, product_list, labels, reorders, not_reordered

        # Enrich with order and product data
        train_df = train_df.merge(orders, how='left')
        train_df = train_df.merge(products, how='left')

        train_df.drop(['eval_set', 'product_name'], axis=1, inplace=True)

        # Move 'reordered' column to last position
        train_df = train_df[[column for column in train_df if column != 'reordered'] + ['reordered']]

        return train_df

    @staticmethod
    def create_test_set(all_products, orders, products):

        order_list = []
        product_list = []

        test_orders = orders[orders.eval_set == "test"]
        for row in tqdm(test_orders.itertuples(), total=len(test_orders), desc="Building Testset"):
            product_list += all_products[row.user_id]
            order_list += [row.order_id] * len(all_products[row.user_id])

        test_df = pd.DataFrame()
        test_df['order_id'] = order_list
        test_df['product_id'] = product_list
        del order_list, product_list

        test_df = test_df.merge(orders, how='left')
        test_df = test_df.merge(products, how='left')

        test_df.drop(['eval_set', 'product_name'], axis=1, inplace=True)

        return test_df

    @staticmethod
    def _create_set(all_products, orders, products, reorders, not_reordered):

        order_list = []
        product_list = []
        labels = []

        # Loop over all train_orders to get a list with all reorders taken on last order with reordered = 1
        # and all other products the user has bought with reordered = 0
        train_orders = orders[orders.eval_set == 'train']

        for row in tqdm(train_orders.itertuples(), total=len(train_orders), desc="Building Trainset"):

            # try/catch faster than 'if row.user_id in reorders.index'
            try:
                # Add all reorders of user and corresponding order_id to list
                product_list += reorders[row.user_id]
                order_list += [row.order_id] * len(reorders[row.user_id])
                labels += [1] * len(reorders[row.user_id])

                # Add all not reordered products of user and corresponding order_id to list
                product_list += not_reordered[row.user_id]
                order_list += [row.order_id] * len(not_reordered[row.user_id])
                labels += [0] * len(not_reordered[row.user_id])

            except:
                # No reorders, so add all products of user and corresponding order_id to list
                product_list += all_products[row.user_id]
                order_list += [row.order_id] * len(all_products[row.user_id])
                labels += [0] * len(all_products[row.user_id])

        train_df = pd.DataFrame()
        train_df['order_id'] = order_list
        train_df['product_id'] = product_list
        train_df['reordered'] = labels
        del order_list, product_list, labels, reorders, not_reordered

        # Enrich with order and product data
        train_df = train_df.merge(orders, how='left')
        train_df = train_df.merge(products, how='left')

        train_df.drop(['eval_set', 'product_name'], axis=1, inplace=True)

        # Move 'reordered' column to last position
        train_df = train_df[[column for column in train_df if column != 'reordered'] + ['reordered']]

        return train_df

