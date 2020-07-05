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


def clear_records(dataset: pd.DataFrame, test_dataset: pd.DataFrame, shops, item_categories, items):
    # "Date" column
    dataset = dataset.drop('date', axis=1).join(items[['item_category_id']], on='item_id')

    for i in ['shop_id', 'item_id', 'date_block_num', 'item_cnt_day', 'item_price']:
        dataset[i] = dataset[i].astype('int16')

    # Shops fix
    for pair in [(0, 57), (1, 58), (10, 11)]:
        dataset.loc[(dataset['shop_id'] == pair[0]), 'shop_id'] = pair[1]
        shops.drop(pair[0], inplace=True)

    dataset['shop_id'] = pd.Categorical(dataset['shop_id'], categories=shops.index)
    dataset['item_id'] = pd.Categorical(dataset['item_id'], categories=items.index)
    dataset['item_category_id'] = pd.Categorical(dataset['item_category_id'], categories=item_categories.index)

    # Outliers
    dataset['item_cnt_day'] = dataset['item_cnt_day'].apply(abs)
    dataset = dataset[(dataset['item_price'] > 0) & (dataset['item_cnt_day'] <= 100)]

    whis = 1
    IQR = dataset['item_price'].quantile(0.75) - dataset['item_price'].quantile(0.25)
    outliers = dataset[dataset['item_price'] > dataset['item_price'].quantile(0.75) + IQR * whis]
    outliers = outliers[(outliers['item_id'].isin(test_dataset['item_id'].unique()) == False) &
                        (outliers['item_price'] >= 10000)]

    print("Records are clear...")

    return dataset.drop(outliers.index)


def get_prices_means(dataset: pd.DataFrame):
    item_mean_prices = dataset.groupby('item_id')['item_price'].mean().fillna(0).astype('int16')
    item_mean_prices.name = 'item_mean_prices'
    shop_mean_prices = dataset.groupby('shop_id')['item_price'].mean().fillna(0).astype('int16')
    shop_mean_prices.name = 'shop_mean_prices'
    cat_mean_prices = dataset.groupby('item_category_id')['item_price'].mean().fillna(0).astype('int16')
    cat_mean_prices.name = 'cat_mean_prices'

    print("Mean prices are found...")

    return item_mean_prices, shop_mean_prices, cat_mean_prices


def generate_zero_sales(dataset, sample_size=800):
    print("Generating zero sales...")

    unique_shops = dataset['shop_id'].unique()
    unique_items = dataset['item_id'].unique()
    zeroes_addiction = []

    for shop_id in unique_shops:
        for date_block_num in range(dataset['date_block_num'].max() + 1):
            temp_items = dataset[(dataset['date_block_num'] == date_block_num) & (dataset['date_block_num'] == shop_id)
                                 ]['item_id'].unique()
            count = 0
            for item in np.random.choice(unique_items, sample_size, replace=False):
                if item not in temp_items:
                    count += 1
                    zeroes_addiction.append([date_block_num, shop_id, item, 0, 0])
                    if count > sample_size:
                        break

    print("Zero sales generated...")

    return zeroes_addiction


def group_records(dataset):
    # Calculate revenue
    data_grouped = dataset.drop('item_category_id', axis=1).append(pd.DataFrame(generate_zero_sales(dataset),
                                                                                columns=['date_block_num', 'shop_id',
                                                                                         'item_id', 'item_price',
                                                                                         'item_cnt_day'],
                                                                                ), ignore_index=True)

    data_grouped['revenue'] = (data_grouped['item_price'] * data_grouped['item_cnt_day']).astype('int16')

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

    united = data_grouped.append(test_dataset, ignore_index=True)

    return united, data_grouped['date_block_num'].max() + 1


def lag_features(full, train_mask, story_len, lag_columns, index_columns=['date_block_num', 'shop_id', 'item_id']):
    print("Calculating time-based features...")

    # Cycle through length
    for k in tqdm([1, 2, 4, 6, 12]):
        gr_copy = (full[full['date_block_num'] < train_mask])[lag_columns + index_columns].astype('int16')

        # Merging is done with use of lagging previous records
        gr_copy['date_block_num'] += k
        gr_copy.rename(columns={column: column + '_lag_{}'.format(k) for column in lag_columns},
                       inplace=True)

        # Merge and deal fill NaNs
        full = pd.merge(full, gr_copy, on=index_columns, how='left')

        # full.fillna({column + '_lag_{}'.format(k): 0 for column in lag_columns},
        #             inplace=True)

    print("Time-based features added...")

    # As "date_block_num" starts with block 0, we have story records for all date blocks after "story_len"
    return full[full['date_block_num'] >= story_len].fillna(0)


def set_categories_features(full, train_mask, category_columns):
    print("Calculating categories' features...")

    # One Hot
    full = full.join(pd.get_dummies(full['shop_id'], prefix='shop'))

    # Count mean values
    sample = full[full['date_block_num'] < train_mask].groupby(['date_block_num'])[
        ['target']].mean().astype('float16')
    sample.rename(columns={'target': 'target_month_mean'}, inplace=True)

    # Mean encoding join
    full = full.join(sample['target_month_mean'], on=['date_block_num'])

    # Count mean values
    sample = full[full['date_block_num'] < train_mask].groupby(['date_block_num', 'shop_id', 'item_category_id'])[
        ['target']].mean().astype('float16')
    sample.rename(columns={'target': 'shop_cat' + '_target_mean'}, inplace=True)

    # Mean encoding join
    full = full.join(sample['shop_cat' + '_target_mean'], on=['date_block_num', 'shop_id', 'item_category_id'])

    for category_column in category_columns:
        # Count mean values
        sample = full[full['date_block_num'] < train_mask].groupby(['date_block_num', category_column])[
            ['target']].mean().astype('float16')
        sample.rename(columns={'target': category_column + '_target_mean'}, inplace=True)

        # Mean encoding join
        full = full.join(sample[category_column + '_target_mean'], on=['date_block_num', category_column])

        # Count mean values
        sample = full[full['date_block_num'] < train_mask].groupby(['date_block_num', category_column])[
            ['revenue']].sum().astype('int32')
        sample.rename(columns={'revenue': category_column + '_revenue_mean'}, inplace=True)

        # Mean encoding join
        full = full.join(sample[category_column + '_revenue_mean'], on=['date_block_num', category_column])

        # Frequency encoding
        sample = full[full['date_block_num'] < train_mask].groupby([
            'date_block_num', category_column]).size() / full[full['date_block_num'] < train_mask].groupby(
            'date_block_num').size().astype('float16')
        sample.name = category_column + '_freq'

        full = full.join(sample, on=['date_block_num', category_column])

    print("Categories' features added...")

    return full


def n_sales(full, story_len):
    print("Calculating number of sales...")

    full['revenue_std'] = full[['revenue_lag_{}'.format(i) for i in [1, 2, 4, 6, 12]]].std(axis=1).astype(
        'float16')
    full['target_std'] = full[['target_lag_{}'.format(i) for i in [1, 2, 4, 6, 12]]].std(axis=1).astype(
        'float16')

    full['zero_sales'] = (full[['revenue_lag_{}'.format(i) for i in [1, 2, 4, 6, 12]]] == 0).sum(axis=1).astype(
        'int8')
    full['nonzero_sales'] = (len([1, 2, 4, 6, 12]) - full['zero_sales']).astype('int8')

    print("Number of sales added...")


def numerical_features(full, numerical_columns, functions):
    print("Calculating numerical features...")

    # Names of functions are presented in form of suffixes
    for column in numerical_columns:
        for key in functions.keys():
            full[column + key] = full[column].apply(functions[key]).astype('float32')

    print("Numerical features added...")


def different_items_sold(full):
    grouped_data = full.groupby('shop_id')['item_id'].nunique()
    grouped_data.name = 'unique_items_sold_by_shop'

    return full.join(grouped_data, on='shop_id')


def validation_preparation(full, train_mask, story_len, to_join, join=True):
    category_columns = ['shop_id', 'item_id', 'item_category_id']

    print(full.shape)

    full = set_categories_features(full, train_mask, category_columns)

    print(full.shape)

    if join:
        for i in to_join:
            full = full.join(i, on=i.reset_index().columns[0])

    full = different_items_sold(full)

    print(full.shape)

    full = lag_features(full, train_mask, story_len,
                        lag_columns=['shop_cat_target_mean', 'target_month_mean', 'target', 'revenue', 'unique_items_sold_by_shop'] + [
                            i + '_freq' for i in category_columns] + [i + '_target_mean' for i in category_columns] + [
                            i + '_revenue_mean' for i in category_columns])

    print(full.shape)

    n_sales(full, story_len)

    '''
    numerical_columns = ['shop_id_freq_lag_{}'.format(i) for i in [1, 2, 4, 6, 12]] + [
        'item_id_freq_lag_{}'.format(i) for i in [1, 2, 4, 6, 12]] + [
                            'item_category_id_freq_lag_{}'.format(i) for i in [1, 2, 4, 6, 12]] + [
                            'target_lag_{}'.format(i) for i in [1, 2, 4, 6, 12]]

    functions = {
        '_square': lambda x: x * x,
        '_sqrt': lambda x: -1 if (x < 0) else np.math.sqrt(x)
    }
    
    numerical_features(full, numerical_columns, functions)
    '''

    # "category_columns" consists of nominal values, so they are mostly useless for any model
    full.drop(category_columns + ['shop_cat_target_mean', 'revenue', 'target_month_mean', 'unique_items_sold_by_shop'] + [i + '_freq' for i in category_columns] + [
            i + '_target_mean' for i in category_columns] + [i + '_revenue_mean' for i in category_columns],
            axis=1, inplace=True)

    return full
