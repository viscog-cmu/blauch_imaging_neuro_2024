import numpy as np
import nilearn.decoding
from nilearn.image import index_img, new_img_like
import nibabel as nib
from scipy.io import loadmat
from scipy.stats import zscore
import os
import pdb
import ipdb

from .commons import BIDS_DIR
from .masking import get_full_mask

def neural_laterality(sub, 
                       func_map='uni-faces-vs-words', 
                       mask='vis-on-vs-off',
                       experiment='floc',
                       metric='sum',
                       include_rois=None,
                       exclude_rois=None,
                       mask_neglogp_thresh=3,
                       map_neglogp_thresh=None,
                       return_scores=False,
                       mask_base_fn=None,
                       sm_mm_fwhm=4,
                     ):
    """
    Compute hemispheric laterality index for functional activation within a masked region.
    
    Calculates laterality as (LH - RH)/(LH + RH) where LH/RH are selectivity metrics
    from the left/right hemispheres within the specified ROIs and masks.
    
    Parameters
    ----------
    sub : int
        Subject number
    func_map : str, optional
        Name of functional contrast map, default 'uni-faces-vs-words'
    mask : str, optional
        Name of mask contrast map, default 'vis-on-vs-off'
    experiment : str, optional
        Experiment name, default 'floc'
    metric : {'sum', 'mean', 'sumnorm', 'sumnorma', 'peak', 'nvox'}, optional
        Method for computing hemisphere values:
        - sum: Sum of activation values
        - mean: Mean activation value
        - sumnorm: Sum normalized by number of voxels in mask
        - sumnorma: Sum normalized by number of voxels in anatomical ROI
        - peak: Maximum activation value
        - nvox: Number of voxels above threshold
    include_rois : list, optional
        ROIs to include in mask
    exclude_rois : list, optional
        ROIs to exclude from mask
    mask_neglogp_thresh : float, optional
        -log10(p) threshold for mask inclusion, default 3 (p < 0.001)
    map_neglogp_thresh : float, optional
        -log10(p) threshold for activation map
    return_scores : bool, optional
        Whether to return individual hemisphere scores
    mask_base_fn : str, optional
        Base filename for saving hemisphere masks
    sm_mm_fwhm : int, optional
        FWHM of smoothing kernel in mm, default 4
        
    Returns
    -------
    float or tuple
        Laterality index, or tuple of (laterality, lh_score, rh_score) if return_scores=True
        Laterality ranges from -1 (completely right lateralized) to 1 (completely left)
    """

    sm_tag = f'_sm-{sm_mm_fwhm}mm' if sm_mm_fwhm else ''
    func_img =  nib.load(f'{BIDS_DIR}/derivatives/matlab/spm/sub-{sub:02d}/SPM-{experiment}_vol-T1w{sm_tag}/{func_map}.nii')
    
    lh_mask = get_full_mask(sub, include_rois, exclude_rois, 'lh', func_mask=mask, func_thresh=mask_neglogp_thresh)
    rh_mask = get_full_mask(sub, include_rois, exclude_rois, 'rh', func_mask=mask, func_thresh=mask_neglogp_thresh)
    if metric == 'mean' or metric == 'sumnorm':
        lh_nvox = np.sum(lh_mask)
        rh_nvox = np.sum(rh_mask)
    if metric == 'sumnorma':
        lh_a_mask = get_full_mask(sub, include_rois, exclude_rois, 'lh', func_mask=None)
        rh_a_mask = get_full_mask(sub, include_rois, exclude_rois, 'rh', func_mask=None)
        lh_nvox = np.sum(lh_a_mask)
        rh_nvox = np.sum(rh_a_mask)
    
    if map_neglogp_thresh is not None:
        func_neglogp_img =  nib.load(f'{BIDS_DIR}/derivatives/matlab/spm/sub-{sub:02d}/SPM-{experiment}_vol-T1w{sm_tag}/{func_map}_log10p.nii')
        map_mask = func_neglogp_img.get_data() > map_neglogp_thresh
        lh_mask = np.logical_and(lh_mask, map_mask)
        rh_mask = np.logical_and(rh_mask, map_mask)
        
    if mask_base_fn is not None:
        lh_mask_img = new_img_like(func_img, lh_mask)
        rh_mask_img = new_img_like(func_img, rh_mask)
        os.makedirs(f'{BIDS_DIR}/derivatives/python/selectivity_masks/sub-{sub:02d}', exist_ok=True)
        nib.save(lh_mask_img, f'{BIDS_DIR}/derivatives/python/selectivity_masks/sub-{sub:02d}/{mask_base_fn}_lh.nii.gz')
        nib.save(rh_mask_img, f'{BIDS_DIR}/derivatives/python/selectivity_masks/sub-{sub:02d}/{mask_base_fn}_rh.nii.gz')

    func_img = func_img.get_data()
                 
    if metric == 'sum':
        lh_score = np.sum(func_img[lh_mask])
        rh_score = np.sum(func_img[rh_mask])
    elif metric == 'mean':
        if lh_nvox == 0:
            lh_score = 0
        else:
            lh_score = np.sum(func_img[lh_mask])/lh_nvox
        if rh_nvox == 0:
            rh_score = 0
        else:
            rh_score = np.sum(func_img[rh_mask])/rh_nvox
    elif metric == 'sumnorma' or metric == 'sumnorm':
        if lh_nvox+rh_nvox == 0:
            lh_score = 0
            rh_score = 0
        else:
            lh_score = np.sum(func_img[lh_mask])/(lh_nvox+rh_nvox)
            rh_score = np.sum(func_img[rh_mask])/(lh_nvox+rh_nvox)
    elif metric == 'peak':
        if np.sum(lh_mask) > 0:
            lh_score = np.max(func_img[lh_mask])
        else:
            lh_score = 0
        if np.sum(rh_mask) > 0:
            rh_score = np.max(func_img[rh_mask])
        else:
            rh_score = 0
    elif metric == 'nvox':
        if map_neglogp_thresh is None:
            raise ValueError('metric == nvox requires map_neglogp_thresh is not None')
        lh_score = np.sum(lh_mask)
        rh_score = np.sum(rh_mask)
    else: 
        raise NotImplementedError()
            
    if lh_score == 0 and rh_score == 0:
        laterality = np.nan
    else:
        laterality = (lh_score - rh_score)/(lh_score + rh_score)
    
    if return_scores:
        return laterality, lh_score, rh_score
    else:
        return laterality