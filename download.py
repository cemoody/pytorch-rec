from pyhive import presto
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Delete the 'cache' directory from disk to refresh data
seed = 42
mem = joblib.Memory('cache')


@mem.cache
def run_query():
    query = r"""
    SELECT client_id, style_variant_id, age, response_value = '+1' AS is_liked
      FROM erin.styleup_summary
     WHERE date_asked > cast('2017-07-20' as date)
       AND business_line = 'Womens'
    """

    eng = presto.connect("presto-master.vertigo.stitchfix.com", 8889)
    df = pd.read_sql(query, eng)
    df['style_variant_id_code'] = pd.Categorical(df['style_variant_id']).codes
    df['client_id_code'] = pd.Categorical(df['client_id']).codes
    return df


def run():
    df = run_query()

    item = df.style_variant_id_code.values.astype('int64')
    user = df.client_id_code.values.astype('int64')
    uage = df.age.values.astype('float64')
    # Replace missing age with mean age
    idx = np.isfinite(uage)
    uage[~idx] = uage[idx].mean()
    like = df.is_liked.values.astype('float32')
    ret = train_test_split(item, user, uage, like, random_state=seed)
    (train_item, test_item, train_user, test_user,
     train_uage, test_uage, train_like, test_like) = ret

    n_users = len(np.unique(user))
    n_items = len(np.unique(item))
    n_obs = len(item)

    np.savez('train', item=train_item, user=train_user,
             uage=train_uage, like=train_like)
    np.savez('test', item=test_item, user=test_user,
             uage=test_uage, like=test_like)
    np.savez('full', n_users=n_users, n_items=n_items, seed=seed,
             n_obs=n_obs)


if __name__ == '__main__':
    run()
