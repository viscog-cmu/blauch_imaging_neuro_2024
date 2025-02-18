import pandas as pd
import os
import sys
sys.path.append('.')
from func import BIDS_DIR, SUBNUMS

all_subs_df = pd.DataFrame()
for sub in SUBNUMS:
    if os.path.exists(f'{BIDS_DIR}/derivatives/python/ind_diffs/sub-{sub:02d}/VTC.csv'):
        sub_df = pd.read_csv(f'{BIDS_DIR}/derivatives/python/ind_diffs/sub-{sub:02d}/VTC.csv')
        all_subs_df = pd.concat([all_subs_df, sub_df])
    else:
        print(sub)
matching = all_subs_df.sub_1 == all_subs_df.sub_2
all_subs_df['comparison'] = ['Within-subject' if m else 'Between-subject' for m in matching]

all_subs_df.to_csv('data/vtc_ind_diffs.csv', index=False)