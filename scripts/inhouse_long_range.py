import pickle
import os
import numpy as np
import nibabel as nib
from nilearn.surface import load_surf_data

import pickle
from tqdm import tqdm
import sys
sys.path.append('.')

from func import SUBNUMS, BIDS_DIR
from func.neural_laterality import neural_laterality
from func.masking import get_full_mask, get_annot, atlas_to_surf_map, find_overlapping_rois_group

def get_atlas_selectivity(func_map, atlas = 'HCPMMP1', smooth=True, overwrite=False, exclude_rois=None):
    sm_tag = '_sm-4mm' if smooth else ''
    fn = f'{BIDS_DIR}/derivatives/python/atlas_func_stats/{func_map}_{atlas}{sm_tag}.pkl'
    annot = get_annot(atlas)
    surf_maps = []
    all_sums = []
    if os.path.exists(fn) and not overwrite:
        with open(fn, 'rb') as f:
            all_sums, surf_maps, inds = pickle.load(f)
    else:
        for sub in tqdm(SUBNUMS):
            try:
                atlas_img = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/T1w.{atlas}+aseg.nii.gz').get_fdata()
            except:
                atlas_img = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/T1w.{atlas}+aseg.nii').get_fdata()
            if exclude_rois is not None:
                for roi in exclude_rois:
                    atlas_img[atlas_img == roi] = 0
            func_img = nib.load(f'{BIDS_DIR}/derivatives/matlab/spm/sub-{sub:02d}/SPM-floc_vol-T1w{sm_tag}/{func_map}.nii').get_fdata()
            inds = np.unique(atlas_img) 
            inds = inds[inds>1000]
            inds = inds[inds != 2000]
            sums = []
            for ind in inds:
                roi_dat = func_img[atlas_img == ind]
                sums.append(np.sum(roi_dat[roi_dat>0])/len(roi_dat))
            all_sums.append(sums)
            surf_maps.append(atlas_to_surf_map(sums, inds, annot))
        all_sums, surf_maps = np.array(all_sums), np.array(surf_maps)
        with open(fn, 'wb') as f:
            pickle.dump((all_sums, surf_maps, inds), f)
    return all_sums, surf_maps, inds

def get_roi_selectivity(func_map, func_mask, include_rois, exclude_rois=[], smooth=False, **kwargs):
    lhs, rhs, lateralitys = [], [], []
    for sub in tqdm(SUBNUMS):
        laterality, lh, rh = neural_laterality(sub, func_map, func_mask, include_rois=include_rois, exclude_rois=exclude_rois, mask_neglogp_thresh=3, map_neglogp_thresh=0, sm_mm_fwhm=4 if smooth else 0, return_scores=True, **kwargs)
        lhs.append(lh)
        rhs.append(rh)
        lateralitys.append(laterality)
    lhs = np.array(lhs)
    rhs = np.array(rhs)
    lateralitys = np.array(lateralitys)
    return lhs, rhs, lateralitys

if __name__ == '__main__':

    # atlas = 'HCPMMP1'
    atlas = '1000Parcels_Yan2023_Yeo2011_17Networks'

    data = {}
    for subset in ['even', 'odd']:
        data[subset] = {}
        contrast_maps = {
            'faces_vs_objects': 'uni-faces-vs-objects-comb',
            'text_vs_objects': 'uni-characters-vs-objects-comb',
        }
        for categ, func_map in contrast_maps.items():
            all_sums, surf_maps, inds = get_atlas_selectivity(f'{func_map}-{subset}', atlas = atlas, smooth=True, overwrite=True, exclude_rois=None)
            rois_per_hemi = len(inds)//2
            laterality = (all_sums[:,:rois_per_hemi] - all_sums[:,rois_per_hemi:])/(all_sums[:,:rois_per_hemi] + all_sums[:,rois_per_hemi:])
            laterality[np.isinf(laterality)] = 0
            laterality[np.isnan(laterality)] = 0
            full_laterality = np.hstack((laterality, -laterality))
            data[subset][categ] = {}
            data[subset][categ]['all_sums'] = all_sums
            data[subset][categ]['surf_maps'] = surf_maps
            data[subset][categ]['inds'] = inds
            data[subset][categ]['laterality'] = laterality
            data[subset][categ]['full_laterality'] = full_laterality
            lh, rh, laterality = get_roi_selectivity(f'{func_map}-{subset}', None, ['fusiform', 'inferiortemporal'], [], smooth=True)
            data[subset][categ]['VTC_lh'] = lh
            data[subset][categ]['VTC_rh'] = rh
            data[subset][categ]['VTC_laterality'] = laterality

    with open(f'data/inhouse_long_range_laterality_{atlas}.pkl', 'wb') as f:
        pickle.dump(data, f)

exclude_rois = find_overlapping_rois_group(atlas, nvox_thresh=5)

with open(f'data/{atlas}_exclude_rois.pkl', 'wb') as f:
    pickle.dump(exclude_rois, f)