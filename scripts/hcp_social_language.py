import pandas as pd
import sys
import numpy as np
from scipy.stats import spearmanr
sys.path.append('.')
from func.masking import RAJIMEHR_ROIS
from func.hcp import clean_contrast_name
from func.hcp_plotting_funs import get_sums, get_roi_from_atlas

handedness = 'R'
hand_thresh = 40

all_rois = RAJIMEHR_ROIS.keys()
sums, inds = get_sums(atlas='rajimehr', mask_tag='', experiments=['SOCIAL', 'LANGUAGE'], hand_thresh=hand_thresh)

df = {'roi':[], 'r':[], 'p':[], 'laterality_style':[], 'dof':[]}

for ii, (laterality_style, laterality_name) in enumerate([('laterality', '(L-R)/(L+R)'), ('laterality_diff', 'L-R')]):
    for roi in all_rois:
        social, _ = get_roi_from_atlas(sums, 'TOM-RANDOM', 'SOCIAL', handedness, [roi], laterality_style, atlas='rajimehr')
        language, _ = get_roi_from_atlas(sums, 'STORY', 'LANGUAGE', handedness, [roi], laterality_style, atlas='rajimehr')
        non_nan_inds = np.logical_not(np.isnan(social) | np.isnan(language))
        r,p = spearmanr(social[non_nan_inds], language[non_nan_inds])

        df['roi'].append(roi)
        df['r'].append(r)
        df['p'].append(p)
        df['laterality_style'].append(laterality_style)
        df['dof'].append(np.sum(non_nan_inds))

                
df = pd.DataFrame(df)
df.to_csv(f'data/hcp_social_language_laterality_hand-R.csv', index=False)
