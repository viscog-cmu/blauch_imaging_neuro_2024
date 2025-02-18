# quantify differences in individual patterns of VTC selectivity

import numpy as np
import os
import sys
from tqdm import tqdm
import argparse
import pandas as pd
import nibabel as nib
sys.path.append('.')
from func.pycortex_utils import get_vtc_masks
from func import SUBNUMS, BIDS_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, help='Subject ID')
args = parser.parse_args()

roi_dict = {
    'VTC': {'include': ['inferiortemporal', 'fusiform'], 'exclude':None}, 
}
func_maps = {
    'text': 'uni-characters-vs-others-comb',
    'faces': 'uni-faces-vs-others-comb', 
    'objects': 'uni-objects-vs-others-comb', 
}

sub_1 = args.sub

df_between_sub = {'sub_1':[], 'sub_2':[], 'hemi':[], 'func_map':[], 'r':[], 'dice':[], 'smoothed':[]}
for hemi in ['lh', 'rh']:
    hemisphere = hemi[0].upper()
    for func_map, uni_map in func_maps.items():
        for smooth in [True]:
            sm_tag = '_sm-4mm' if smooth else ''
            map_1 = nib.load(f'{BIDS_DIR}/derivatives/matlab/spm/sub-{sub_1:02d}/SPM-floc_vol-T1w{sm_tag}/{uni_map}-even_log10p_fsaverage_{hemisphere}.gii').darrays[0].data
            mask_1 = get_vtc_masks(subject='fsaverage', concatenate=False)[hemisphere]
            roi_1 = map_1[mask_1]
            for sub_2 in tqdm(SUBNUMS):
                map_2 = nib.load(f'{BIDS_DIR}/derivatives/matlab/spm/sub-{sub_2:02d}/SPM-floc_vol-T1w{sm_tag}/{uni_map}-odd_log10p_fsaverage_{hemisphere}.gii').darrays[0].data
                mask_2 = get_vtc_masks(subject='fsaverage', concatenate=False)[hemisphere]
                roi_2 = map_2[mask_2]

                r = np.corrcoef(map_1, map_2)[0,1]

                # threshold maps and compute dice coefficient
                roi_1_masked = roi_1 > 3
                roi_2_masked = roi_2 > 3
                dice = np.sum(roi_1_masked & roi_2_masked) * 2.0 / (np.sum(roi_1_masked) + np.sum(roi_2_masked))

                df_between_sub['r'].append(r)
                df_between_sub['dice'].append(dice)
                df_between_sub['sub_1'].append(sub_1)
                df_between_sub['sub_2'].append(sub_2)
                df_between_sub['hemi'].append(hemi)
                df_between_sub['func_map'].append(func_map)
                df_between_sub['smoothed'].append(smooth)

df_between_sub = pd.DataFrame(df_between_sub)
os.makedirs(f'{BIDS_DIR}/derivatives/python/ind_diffs/sub-{sub_1:02d}', exist_ok=True)
df_between_sub.to_csv(f'{BIDS_DIR}/derivatives/python/ind_diffs/sub-{sub_1:02d}/VTC.csv', index=False)