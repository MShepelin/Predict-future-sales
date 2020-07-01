import pandas as pd
import numpy as np
from tqdm import tqdm


# Just useful function to see if some value has changes with respect to group of other columns
# Reminder: this function is a bit heavy
def has_changes_column(data, group_columns, check_column):
    group_data = data.groupby(group_columns)[[check_column]].nunique().reset_index()
    group_data = group_data[group_data[check_column] != 1]
    group_data['has_changes'] = True
    group_data = group_data[group_columns + ['has_changes']]
    return pd.merge(data, group_data, on=group_columns, how='left')['has_changes'].fillna(False)


def clear_records(dataset: pd.DataFrame, test_dataset: pd.DataFrame, shops, inplace=False):
    if not inplace:
        dataset = dataset.copy()

    # "Date" column
    dataset.drop('date', axis=1, inplace=True)

    # Shops fix
    for pair in [(0, 57), (1, 58), (10, 11)]:
        dataset.loc[(dataset['shop_id'] == pair[0]), 'shop_id'] = pair[1]
        shops.drop(pair[0], inplace=True)

    # Outliers
    dataset['item_cnt_day'] = dataset['item_cnt_day'].apply(abs)
    dataset = dataset[(dataset['item_price'] > 0) & (dataset['item_cnt_day'] <= 700)]

    whis = 1.5
    IQR = dataset['item_price'].quantile(0.75) - dataset['item_price'].quantile(0.25)
    outliers = dataset[dataset['item_price'] > dataset['item_price'].quantile(0.75) + IQR * whis]
    outliers = outliers[(outliers['item_id'].isin(test_dataset['item_id'].unique()) == False) &
                        (outliers['item_price'] >= 45000)]

    dataset.drop(outliers.index, inplace=True)

    print("Records are clear...")

    if not inplace:
        return dataset, shops


def get_prices_means(dataset: pd.DataFrame):
    item_mean_prices = dataset.groupby('item_id')['item_price'].mean().astype('int32')
    item_mean_prices.name = 'item_mean_prices'
    shop_mean_prices = dataset.groupby('shop_id')['item_price'].mean().astype('int32')
    shop_mean_prices.name = 'shop_mean_prices'
    cat_mean_prices = dataset.groupby('item_category_id')['item_price'].mean().astype('int32')
    cat_mean_prices.name = 'cat_mean_prices'

    print("Mean prices are found...")

    return item_mean_prices, shop_mean_prices, cat_mean_prices


def generate_zero_sales(dataset, sample_size=200):
    unique_shops = dataset['shop_id'].unique()
    unique_items = dataset['item_id'].unique()
    zeroes_addiction = []

    for shop_id in unique_shops:
        for date_block_num in range(dataset['date_block_num'].max() + 1):
            for item in np.random.choice(unique_items, sample_size, replace=False):
                zeroes_addiction.append([date_block_num, shop_id, 0, item, 0])

    print("Zero sales generated...")

    return zeroes_addiction


def group_records(dataset):
    # Calculate revenue
    data_grouped = dataset.append(pd.DataFrame(generate_zero_sales(dataset), columns=dataset.columns))
    data_grouped['revenue'] = (data_grouped['item_price'] * data_grouped['item_cnt_day']).astype('float32')
    data_grouped.drop('item_price', axis=1, inplace=True)

    # Sum up day sales for each month
    data_grouped = data_grouped.groupby(['date_block_num', 'shop_id', 'item_id'])[
        ['item_cnt_day', 'revenue']].sum().reset_index()
    data_grouped.rename(columns={'item_cnt_day': 'target'}, inplace=True)
    data_grouped['target'].clip(0, 20, inplace=True)

    print("Records collected by month...")

    return data_grouped


def union(data_grouped, test_dataset):
    test_dataset['date_block_num'] = data_grouped['date_block_num'].max() + 1
    test_dataset['revenue'] = np.nan
    test_dataset['target'] = np.nan

    test_dataset = test_dataset[data_grouped.columns]

    united = data_grouped.append(test_dataset)

    return united, data_grouped['date_block_num'].max() + 1


def lag_features(full, train_mask, story_len, lag_columns=['target', 'revenue'], nan_fill={'target': 0, 'revenue': 0},
                           index_columns=['date_block_num', 'shop_id', 'item_id']):
    # Cycle through length
    for k in tqdm(range(1, story_len + 1)):
        gr_copy = full[lag_columns + index_columns].copy()

        # Merging is done with use of lagging previous records
        gr_copy['date_block_num'] += k
        for column in lag_columns:
            gr_copy.rename(columns={column: column + '_lag_{}'.format(k)},
                           inplace=True)

        # Merge and deal fill NaNs
        full = pd.merge(full, gr_copy, on=index_columns, how='left')
        for column in lag_columns:
            full[full['date_block_num'] < train_mask] = full[full['date_block_num'] < train_mask].fillna({column + '_lag_{}'.format(k): nan_fill[column]})

    print("Time-based features added...")

    # As "date_block_num" starts with block 0, we have story records for all date blocks after "story_len"
    return full[full['date_block_num'] >= story_len]


def set_categories_features(full, train_mask, category_columns):
    # One Hot
    full = full.join(pd.get_dummies(full['item_category_id'], prefix='category').astype('int8'))

    # On validation another scheme should be used!
    for category_column in category_columns:
        # Count mean values
        sample = full[full['date_block_num'] < train_mask].groupby(category_column)[['target']].mean()
        sample.rename(columns={'target': category_column + '_target_mean'}, inplace=True)

        # Mean encoding join
        full = full.join(sample[category_column + '_target_mean'], on=category_column)

        # Frequency encoding
        sample = full[full['date_block_num'] < train_mask].groupby(category_column).size() / len(full[full['date_block_num'] < train_mask])
        sample.name = category_column + '_freq'

        full = full.join(sample, on=category_column)

    print("Categories' features added...")

    return full


def n_sales(full, story_len):
    full['revenue_std'] = full[['revenue_lag_{}'.format(i) for i in range(1, story_len + 1)]].std(axis=1).astype('float32')
    full['target_std'] = full[['target_lag_{}'.format(i) for i in range(1, story_len + 1)]].std(axis=1).astype('float32')

    full['zero_sales'] = (full[['revenue_lag_{}'.format(i) for i in range(1, story_len + 1)]] == 0).sum(axis=1).astype('float32')
    full['nonzero_sales'] = (story_len - full['zero_sales']).astype('float32')

    print("Number of sales added...")


def numerical_features(full, numerical_columns, functions):
    # Names of functions are presented in form of suffixes
    for column in numerical_columns:
        for key in functions.keys():
            full[column + key] = full[column].apply(functions[key]).astype('float32')

    print("Numerical features added...")


def validation_preparation(full, train_mask, story_len, to_join):
    full = lag_features(full, train_mask, story_len)

    category_columns = ['shop_id', 'item_id', 'item_category_id']
    full = set_categories_features(full, train_mask, category_columns)

    n_sales(full, story_len)

    numerical_columns = ['revenue', 'shop_id_target_mean', 'shop_id_freq', 'item_id_target_mean',
                         'item_id_freq', 'item_category_id_target_mean', 'item_category_id_freq',
                         'target_lag_1', 'revenue_lag_1']

    functions = {
        '_square': lambda x: x * x,
        '_sqrt': lambda x: -1 if (x < 0) else np.math.sqrt(x),
        '_cube': lambda x: x * x * x
    }

    numerical_features(full, numerical_columns, functions)

    for i in to_join:
        full = full.join(i, on=i.reset_index().columns[0])

    # "category_columns" consists of nominal values, so they are mostly useless for any model
    full.drop(category_columns + ['revenue'], axis=1, inplace=True)

    return full