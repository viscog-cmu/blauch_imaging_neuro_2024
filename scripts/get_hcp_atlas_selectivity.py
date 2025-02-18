import sys
import itertools
sys.path.append('.')
from func import hcp
import os
import numpy as np
from func.hcp_conversion import hcp_to_fsaverage
from func import HCP_DIR

sel_thresh = 0
space = ''
overwrite = True
exps_by_atlas = {
    'rajimehr': ['WM', 'EMOTION','SOCIAL', 'LANGUAGE'],
    'HCPMMP1': ['WM', 'EMOTION', 'SOCIAL', 'LANGUAGE'],
}

for atlas, exps in exps_by_atlas.items():
    for exp in exps:
        contrasts = hcp.EXP_CONTRASTS[exp]
        these_sums, inds, subs = hcp.get_atlas_selectivity(exp, contrasts, atlas=atlas, overwrite=overwrite, handedness='B', 
                                                        sel_thresh=sel_thresh, space=space)