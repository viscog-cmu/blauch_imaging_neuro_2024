import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.append('.')

from func.neural_laterality import neural_laterality
from func import SUBNUMS
from func.utils import reject_df_outliers

# set some arguments
mask_lang = False # whether to mask language regions with a visual stimulus-responsive mask (p<0.1)
mask_vis = True # whether to mask visual regions with a language stimulus-responsive mask (p<0.001)
smooth = True
metrics = ['sum', 'nvox', 'peak']
subsets = ['_even', '_odd', '']
sm_tag = '_nosmooth' if not smooth else ''

# ------------------------------------------------------VTC analyses----------------------------------------------------
func_maps={
    'text': 'uni-characters-vs-others-comb',
    'faces': 'uni-faces-vs-others-comb',
    'objects': 'uni-objects-vs-others-comb',
    'text_vs_objects': 'uni-characters-vs-objects-comb',
    'faces_vs_objects': 'uni-faces-vs-objects-comb',
    'text_vs_faces': 'uni-characters-vs-faces-comb',
    'objects_vs_faces': 'uni-objects-vs-faces-comb',
    'faces_vs_text': 'uni-faces-vs-text-comb',
    'objects_vs_text': 'uni-objects-vs-text-comb',
    'text_vs_fixation': 'uni-characters-vs-fixation',
    'faces_vs_fixation': 'uni-faces-vs-fixation',
    'objects_vs_fixation': 'uni-objects-vs-fixation',
}
glasser_rois={
    'VTC': {'include': ['inferiortemporal', 'fusiform'], 'exclude':None},
    'LO':  {'include': ['lateraloccipital'], 'exclude': ['V1', 'V2', 'V3']},
    }
func_mask={'VTC':{'mask':'vis-on-vs-off' if mask_vis else None, 'mask_neglogp_thresh':3},
           'LO':{'mask':'vis-on-vs-off' if mask_vis else None, 'mask_neglogp_thresh':3},
}
sm_mm_fwhm=4 if smooth else 0

results = defaultdict(list)
for ii, sub in enumerate(tqdm(SUBNUMS)):
    sub_id = f'{sub:02d}'
    results['subnum'].append(sub)

    # functional results
    for contrast, func_map in func_maps.items():
        for roi in glasser_rois:
            for subset in subsets:
                subset_ = subset.replace('_', '-')
                for metric in metrics:
                    try:
                        laterality, lh_score, rh_score = neural_laterality(sub, 
                                                                        func_map=func_map+subset_,
                                                                        include_rois=glasser_rois[roi]['include'],
                                                                        exclude_rois=glasser_rois[roi]['exclude'],
                                                                        metric=metric,
                                                                        mask=func_mask[roi]['mask']+subset_ if func_mask[roi]['mask'] is not None else None,
                                                                        mask_neglogp_thresh=func_mask[roi]['mask_neglogp_thresh'],
                                                                        map_neglogp_thresh=3 if metric =='nvox' else 0,
                                                                        sm_mm_fwhm=sm_mm_fwhm,
                                                                        return_scores=True)
                        
                        results[f'{roi}_{contrast}_{metric}_lh{subset}'].append(lh_score)
                        results[f'{roi}_{contrast}_{metric}_rh{subset}'].append(rh_score)
                        results[f'{roi}_{contrast}_{metric}_laterality{subset}'].append(laterality)
                        
                    except Exception as e:
                        results[f'{roi}_{contrast}_{metric}_lh{subset}'].append(np.nan)
                        results[f'{roi}_{contrast}_{metric}_rh{subset}'].append(np.nan)
                        results[f'{roi}_{contrast}_{metric}_laterality{subset}'].append(np.nan)
                        print(roi)
                        print(e)
                        print('continuing...')

results = pd.DataFrame(results)
results.to_csv(f'data/inhouse_fmri_mv-{mask_vis}{sm_tag}.tsv', sep='\t')


# ----------------------------------------------------text network analyses-------------------------------------------

func_maps={
    'text': 'uni-characters-vs-others-comb',
    'text_vs_objects': 'uni-characters-vs-objects-comb',
    'text_vs_fixation': 'uni-characters-vs-fixation',
}
glasser_rois={
              'VTC': {'include': ['inferiortemporal', 'fusiform'], 'exclude':None}, 
              'EVC': {'include': ['V1', 'V2', 'V3'], 'exclude':None},
              'IPS': {'include': ['IP0', 'IP1', 'IP2', 'IPS1'], 'exclude':None},
              "IFG": {'include': ['44','IFJa','6r','IFSp'], 'exclude':None}, 
              "IFGorb": {'include': ['IFSa', '45'], 'exclude':None},
              'STSG':{'include': ['STSdp','TPOJ1','A5','STV'], 'exclude':None}, 
              'PCG': {'include': ['55b','PEF'], 'exclude':None},
                }
func_mask={
            'VTC':{'mask':'vis-on-vs-off' if mask_vis else None, 'mask_neglogp_thresh':3}, 
            'EVC':{'mask':'vis-on-vs-off' if mask_vis else None, 'mask_neglogp_thresh':3},
            'IPS':{'mask':'vis-on-vs-off' if mask_vis else None, 'mask_neglogp_thresh':3},
           "IFG": {'mask':'vis-on-vs-off' if mask_lang else None, 'mask_neglogp_thresh':1},
           "IFGorb": {'mask':'vis-on-vs-off' if mask_lang else None, 'mask_neglogp_thresh':1},
           'STSG':{'mask':'vis-on-vs-off' if mask_lang else None, 'mask_neglogp_thresh':1},
           'PCG':{'mask':'vis-on-vs-off' if mask_lang else None, 'mask_neglogp_thresh':1},
            }
sm_mm_fwhm=4 if smooth else 0

results = defaultdict(list)
for ii, sub in enumerate(tqdm(SUBNUMS)):
    sub_id = f'{sub:02d}'
    results['subnum'].append(sub)

    # functional results
    for contrast, func_map in func_maps.items():
        for roi in glasser_rois:
            for subset in subsets:
                subset_ = subset.replace('_', '-')
                for metric in metrics:
                    try:
                        laterality, lh_score, rh_score = neural_laterality(sub, 
                                                                        func_map=func_map+subset_,
                                                                        include_rois=glasser_rois[roi]['include'],
                                                                        exclude_rois=glasser_rois[roi]['exclude'],
                                                                        metric=metric,
                                                                        mask=func_mask[roi]['mask']+subset_ if func_mask[roi]['mask'] is not None else None,
                                                                        mask_neglogp_thresh=func_mask[roi]['mask_neglogp_thresh'],
                                                                        map_neglogp_thresh=3 if metric =='nvox' else 0,
                                                                        sm_mm_fwhm=sm_mm_fwhm,
                                                                        return_scores=True)
                        
                        results[f'{roi}_{contrast}_{metric}_lh{subset}'].append(lh_score)
                        results[f'{roi}_{contrast}_{metric}_rh{subset}'].append(rh_score)
                        results[f'{roi}_{contrast}_{metric}_laterality{subset}'].append(laterality)
                        
                    except Exception as e:
                        results[f'{roi}_{contrast}_{metric}_lh{subset}'].append(np.nan)
                        results[f'{roi}_{contrast}_{metric}_rh{subset}'].append(np.nan)
                        results[f'{roi}_{contrast}_{metric}_laterality{subset}'].append(np.nan)
                        print(roi)
                        print(e)
                        print('continuing...')

results = pd.DataFrame(results)
results.to_csv(f'data/inhouse_fmri_text_network_mv-{mask_vis}{sm_tag}.tsv', sep='\t')