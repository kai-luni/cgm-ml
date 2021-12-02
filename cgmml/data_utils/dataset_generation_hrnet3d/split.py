import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv('metadata.csv')

    # Split
    scan_id_series = df.scan_id.unique()
    train_set_tmp, test_set = train_test_split(scan_id_series, test_size=0.1, random_state=1)
    train_set, val_set = train_test_split(train_set_tmp, test_size=0.1, random_state=1)

    print('Unique scan_ids:', scan_id_series.shape)
    print('Unique Train/val/test scans:', train_set.shape, val_set.shape, test_set.shape)

    # Create new dataframe column and fill according to the split
    df.loc[:, 'dset_split'] = np.NaN
    df.loc[df.scan_id.isin(train_set), 'dset_split'] = 'train'
    df.loc[df.scan_id.isin(val_set), 'dset_split'] = 'val'
    df.loc[df.scan_id.isin(test_set), 'dset_split'] = 'test'

    print('Train/val/test artifacts:',
          df[df['dset_split'] == 'train'].shape,
          df[df['dset_split'] == 'val'].shape,
          df[df['dset_split'] == 'test'].shape)
    assert np.all(df['dset_split'].isin(['train', 'val', 'test']))

    df.to_csv('metadata_with_dset_split.csv')
    del df
