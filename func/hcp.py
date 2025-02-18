from multiprocessing.sharedctypes import Value
import numpy as np
import nibabel as nib
nib.imageglobals.logger.level = 40
import glob
import os
import pandas as pd
from tqdm import tqdm
import copy
import pickle
import hcp_utils
from . import masking
from . import HCP_DIR
from .hcp_conversion import fslr_fill_cifti, hcp_to_fsaverage
from neuromaps import transforms as neuro_transforms
from typing import Literal

RESTRICTED_DATA_PATH = f'{HCP_DIR}/hcp_restricted_data.csv'
UNRESTRICTED_DATA_PATH = f'{HCP_DIR}/hcp_unrestricted_data.csv'
restricted_df = pd.read_csv(RESTRICTED_DATA_PATH) 
SUBJECTS = [str(sub) for sub in restricted_df['Subject']]

EXP_CONTRASTS = {
    'LANGUAGE': ['MATH', 'STORY', 'MATH-STORY', 'STORY-MATH'],
    'WM': ['CUSTOM:FACE-PLACE', 'CUSTOM:TOOL-PLACE', 'CUSTOM:BODY-TOOL', 'CUSTOM:TOOL-BODY', 'CUSTOM:BODY-PLACE', 
           'CUSTOM:TOOL-FACE', 'CUSTOM:BODY-FACE', 'CUSTOM:FACE+BODY-TOOL',
           'FACE-AVG', 'BODY-AVG', 'PLACE-AVG', 'TOOL-AVG', '2BK-0BK', '0BK-2BK', 'FACE', 'TOOL', 'BODY', 'PLACE'],
    'SOCIAL': ['RANDOM', 'TOM', 'TOM-RANDOM'],
    'EMOTION': ['FACES', 'SHAPES', 'FACES-SHAPES', 'neg_FACES', 'neg_SHAPES', 'SHAPES-FACES'],
    'RELATIONAL': ['MATCH', 'REL', 'MATCH-REL', 'REL-MATCH', 'neg_MATCH', 'neg_REL'],
}   

ROIS_DICT = {
           'VTC': ['V8', 'FFC', 'PIT', 'PHA1', 'PHA2', 'PHA3', 'TE2p', 'TF', 'PH', 'VMV3', 'VVC', 'VMV1', 'VMV2'],
           'FFC': ['FFC'],
}

def _get_cifti_roi_mask(sub, roi_name, hemisphere='both', atlas='aparc'):
    if atlas == 'aparc':
        raise NotImplementedError()
        atlas_dict = masking.APARC_ROIS
    elif atlas == 'glasser':
        atlas_dict = masking.GLASSER_ROIS
        atlas = hcp_utils.mmp['map_all']
    ind = atlas_dict[roi_name]
    if hemisphere == 'lh':
        mask = atlas == ind
    elif hemisphere == 'rh':
        mask = atlas == ind + len(atlas_dict)
    elif hemisphere == 'both':
        mask = np.logical_or(atlas == ind, atlas==len(atlas_dict)+ind)
    return mask

def get_single_roi_mask(sub, roi_name, hemisphere):
    if roi_name in masking.GLASSER_ROIS.keys():
        atlas = 'glasser'
    elif roi_name in masking.APARC_ROIS.keys():
        atlas = 'aparc'
    else:
        raise ValueError(f'{roi_name} not in any known atlases')
    return _get_cifti_roi_mask(sub, roi_name, hemisphere, atlas)

def get_roi_mask(sub, rois_to_include, rois_to_exclude, hemisphere):
    if rois_to_include is None or len(rois_to_include) == 0:
        raise ValueError('requires at least one ROI to include')
        
    inclusion = []
    for roi in rois_to_include:
        inclusion.append(get_single_roi_mask(sub, roi, hemisphere))
    inclusion = np.logical_or.reduce([mask for mask in inclusion])
    if rois_to_exclude is None or len(rois_to_exclude) == 0:
        pass
    else:
        exclusion = []
        for roi in rois_to_exclude:
            exclusion.append(get_single_roi_mask(sub, roi, hemisphere))
        exclusion = np.logical_or.reduce([mask for mask in exclusion])
        inclusion[exclusion] = 0
    return inclusion

def neural_laterality(subject, 
                    exp='WM', 
                    contrast='0BK_FACE',
                    metric='sum',
                    include_rois=['V8', 'FFC', 'PIT', 'PHA1', 'PHA2', 'PHA3', 'TE2p', 'TF', 'PH', 'VMV3', 'VVC', 'VMV1', 'VMV2'],
                    exclude_rois=None,
                    map_thresh=None,
                    return_scores=False,
                    data_file=None,
):
    """
    Compute laterality of selectivity within a masked region.
    
    Parameters
    ----------
    subject : str
        Subject ID
    exp : str, optional
        Experiment name, default 'WM'
    contrast : str, optional
        Contrast name, default '0BK_FACE'
    metric : {'sum', 'mean', 'peak', 'nvox'}, optional
        Method for computing laterality, default 'sum'
    include_rois : list, optional
        ROIs to include in mask
    exclude_rois : list, optional
        ROIs to exclude from mask
    map_thresh : float, optional
        Threshold for statistical map
    return_scores : bool, optional
        Whether to return individual hemisphere scores
    data_file : str, optional
        Pre-loaded data file path
        
    Returns
    -------
    float or tuple
        Laterality index, or tuple of (laterality, lh_score, rh_score) if return_scores=True
    """
    lh_mask = get_roi_mask(subject, include_rois, exclude_rois, 'lh')
    rh_mask = get_roi_mask(subject, include_rois, exclude_rois, 'rh')

    if data_file is None:
        func_map = nib.load(f'{HCP_DIR}/subjects/{subject}/MNINonLinear/Results/tfMRI_{exp}/tfMRI_{exp}_hp200_s2_level2.feat/{subject}_tfMRI_{exp}_level2_hp200_s2.dscalar.nii')
    else:
        func_map = data_file

    contrasts = get_contrasts(exp) # didn't download them for all subjects, but they are all identical
    func_map = func_map.get_fdata()[contrasts.index(contrast)]

    if map_thresh is not None:
        if map_thresh != 0:
            assert metric == 'nvox'
        map_mask = func_map > map_thresh
        lh_amask = lh_mask.copy()
        rh_amask = rh_mask.copy()
        lh_mask = np.logical_and(lh_mask, map_mask)
        rh_mask = np.logical_and(rh_mask, map_mask)

    if metric == 'sum':
        lh_score = np.sum(func_map[lh_mask])
        rh_score = np.sum(func_map[rh_mask])
    elif metric == 'sumnorma':
        lh_score = np.sum(func_map[lh_mask])/(np.sum(lh_amask)+np.sum(rh_amask))
        rh_score = np.sum(func_map[rh_mask])/(np.sum(lh_amask)+np.sum(rh_amask))
    elif metric == 'mean':
        lh_score = np.mean(func_map[lh_mask])
        rh_score = np.mean(func_map[rh_mask])
    elif metric == 'peak':
        if np.sum(lh_mask) > 0:
            lh_score = np.max(func_map[lh_mask])
        else:
            lh_score = 0
        if np.sum(rh_mask) > 0:
            rh_score = np.max(func_map[rh_mask])
        else:
            rh_score = 0
    elif metric == 'nvox':
        if map_thresh is None:
            raise ValueError('metric == nvox requires map_neglogp_thresh is not None')
        lh_score = np.sum(lh_mask)
        rh_score = np.sum(rh_mask)
    else:
        raise NotImplementedError()

    try:
        laterality = (rh_score - lh_score)/(lh_score + rh_score)
    except:
        laterality = np.nan
    
    if return_scores:
        return laterality, lh_score, rh_score
    else:
        return laterality

def get_contrasts(exp):
    """
    Get list of available contrasts for an experiment.
    
    Parameters
    ----------
    exp : str
        Experiment name
        
    Returns
    -------
    list
        List of contrast names
    """
    try:
        with open(f'{HCP_DIR}/subjects/100206/MNINonLinear/Results/tfMRI_{exp}/tfMRI_{exp}_hp200_s2_level2.feat/Contrasts.txt', 'r') as f:
            contrasts = f.readlines()
    except:
        with open(f'{HCP_DIR}/func_files/{exp}_contrasts.txt', 'r') as f:
            contrasts = f.readlines()
    contrasts = [contrast.replace('\n', '') for contrast in contrasts]
    return contrasts
        
def get_subject_connectome(subject, meas='density', overwrite=False):
    """
    Load structural connectivity data for a subject.
    
    Parameters
    ----------
    subject : str
        Subject ID
    meas : str, optional
        Connectivity measure type, default 'density'
    overwrite : bool, optional
        Whether to overwrite existing data
        
    Returns
    -------
    array-like
        Connectivity matrix
    """
    network_file = glob.glob(f'{HCP_DIR}/proj-6178bcf09538685e5db810ee/bids/derivatives/brainlife.app-networkneuro/sub-{subject}/dwi/*-{meas}_connectivity.csv')
    if len(network_file) > 0:
        connectome = np.array(pd.read_csv(network_file[0], header=None))
    else:
        raise ValueError(f'network data not available for sub-{subject}')
    return connectome

def get_atlas_selectivity(exp, contrasts_to_use, overwrite=False, append=False, atlas='HCPMMP1', 
                          handedness=None, hand_thresh=0, sel_thresh=0, space='',
                          mask_exp=None, mask_contrast=None, mask_thresh=None, mask_reflect=None, mask_tag='',
                         ):
    """
    Get selectivity values for atlas ROIs.
    
    Parameters
    ----------
    exp : str
        Experiment name
    contrasts_to_use : list
        Contrasts to analyze
    overwrite : bool, optional
        Whether to overwrite existing results
    append : bool, optional
        Whether to append to existing results
    atlas : str, optional
        Atlas to use, default 'HCPMMP1'
    handedness : str, optional
        Handedness group to analyze ('L', 'R', or None)
    hand_thresh : float, optional
        Threshold for handedness score
    sel_thresh : float, optional
        Threshold for selectivity
    space : str, optional
        Space for analysis ('' or 'MSMAll')
    mask_exp : str, optional
        Experiment to use for masking
    mask_contrast : str, optional
        Contrast to use for masking
    mask_thresh : float, optional
        Threshold for mask
    mask_reflect : str, optional
        Hemisphere to reflect mask from
    mask_tag : str, optional
        Tag to add to output filename
        
    Returns
    -------
    tuple
        (selectivity values dict, ROI indices, subject IDs)
    """
    stag = f'_stresh-{sel_thresh}' if sel_thresh else ''
    assert space in ['', 'MSMAll']
    fn = f'{HCP_DIR}/derivatives/python/atlas_func_stats/{exp}_{atlas}{stag}{space}{mask_tag}.pkl'
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    contrasts = get_contrasts(exp) 
    surf_maps = []
    all_sums = {contrast:[] for contrast in contrasts_to_use}
    loaded=False
    if os.path.exists(fn) and (not overwrite or append):
        with open(fn, 'rb') as f:
            loaded_sums, inds = pickle.load(f)
        loaded=True
    if not loaded or (loaded and append):
        if atlas == 'HCPMMP1':
            atlas_img = hcp_utils.mmp['map_all']
            inds = np.unique(atlas_img) 
            inds = inds[inds<361]
        elif atlas == 'rajimehr':
            atlas_img = nib.load(f'{HCP_DIR}/rajimehr_lang_atlas.32k_fs_LR.dlabel.nii').get_fdata().squeeze()
            inds = np.unique(atlas_img)
            inds = inds[inds != 0]
        else:
            raise NotImplementedError()
        for ii, sub in enumerate(tqdm(SUBJECTS)):  
            if os.path.exists(f'{HCP_DIR}/func_files/{sub}_tfMRI_{exp}_level2_hp200_s2{space}.dscalar.nii'):
                print(f'{exp} found for sub-{sub}')
                func_img = nib.load(f'{HCP_DIR}/func_files/{sub}_tfMRI_{exp}_level2_hp200_s2{space}.dscalar.nii').get_fdata()
                for contrast in contrasts_to_use:

                    sums = compute_contrast(sub, exp, contrast, atlas_img, inds,
                         out_type='parcellated', 
                         func_img=func_img,
                         mask_exp=mask_exp, mask_contrast=mask_contrast, mask_thresh=mask_thresh, mask_reflect=mask_reflect,
                       )
                    all_sums[contrast].append(sums)
                    
            else:
                for contrast in contrasts_to_use:
                    all_sums[contrast].append(np.nan*np.zeros(len(inds),))
                print(f'{exp} not found for sub-{sub}')
        all_sums = {contrast:np.array(all_sums[contrast]) for contrast in contrasts_to_use}
        
        if loaded:
            all_sums.update(loaded_sums)
            
        with open(fn, 'wb') as f:
            pickle.dump((all_sums, inds), f)
    else:
        all_sums = loaded_sums
          
    if handedness is not None:
        if handedness != 'T':
            assert hand_thresh >= 0, 'handedness threshold must be a positive absolute value unless handedness=B'
        restricted_df = pd.read_csv(RESTRICTED_DATA_PATH) 
        restricted_df['subject'] = [str(sub) for sub in restricted_df['Subject']]

        restricted_df['handedness_mag'] = np.abs(restricted_df['Handedness'])
        restricted_df['handedness_dir'] = [1 if h>0 else -1 for h in restricted_df['Handedness']]
        restricted_df = restricted_df[restricted_df['subject'].isin(SUBJECTS)]
        
        if handedness == 'L':
            sub_inds = restricted_df['Handedness']<-hand_thresh
        elif handedness == 'R' or handedness == 'T':
            sub_inds = restricted_df['Handedness']>hand_thresh
        else:
            raise ValueError()
        sub_inds = np.array(sub_inds)
        subs = np.array(SUBJECTS)[sub_inds]
        for key in all_sums.keys():
            all_sums[key] = all_sums[key][sub_inds]
            
    else:
        subs = SUBJECTS
            
            
    return all_sums, inds, subs


def get_subs_by_handedness(handedness, hand_thresh):
    """
    Get subject IDs filtered by handedness criteria.
    
    Parameters
    ----------
    handedness : str
        Handedness group ('L', 'R', 'T', or None)
    hand_thresh : float
        Threshold for handedness score
        
    Returns
    -------
    tuple
        (filtered subject IDs, boolean mask for filtering)
    """
    if handedness is not None:
        if handedness != 'T':
            assert hand_thresh >= 0, 'handedness threshold must be a positive absolute value unless handedness=T'
        restricted_df = pd.read_csv(RESTRICTED_DATA_PATH) 
        restricted_df['subject'] = [str(sub) for sub in restricted_df['Subject']]

        restricted_df['handedness_mag'] = np.abs(restricted_df['Handedness'])
        restricted_df['handedness_dir'] = [1 if h>0 else -1 for h in restricted_df['Handedness']]
        restricted_df = restricted_df[restricted_df['subject'].isin(SUBJECTS)]
        
        if handedness == 'L':
            sub_inds = restricted_df['Handedness']<-hand_thresh
        elif handedness == 'R' or handedness == 'T':
            sub_inds = restricted_df['Handedness']>hand_thresh
        else:
            raise ValueError()
        sub_inds = np.array(sub_inds)
        subs = np.array(SUBJECTS)[sub_inds]
    else:
        subs = SUBJECTS
        sub_inds = np.arange(len(SUBJECTS))
    return subs, sub_inds


def get_random_subjects(exp, contrasts, subjects, n_subs, space='_MSMAll', smoothing=2, out_fn=None, metric='average', thresh=None):
    """
    Get data from random subset of subjects.
    
    Parameters
    ----------
    exp : str
        Experiment name
    contrasts : list
        Contrasts to analyze
    subjects : array-like
        Pool of subject IDs to sample from
    n_subs : int
        Number of subjects to sample
    space : str, optional
        Space for analysis
    smoothing : int, optional
        Smoothing kernel size
    out_fn : str, optional
        Output filename
    metric : str, optional
        Metric for analysis
    thresh : float, optional
        Threshold for data
        
    Returns
    -------
    dict
        Dictionary of functional maps per contrast
    """
    func_maps = {contrast:[] for contrast in contrasts}
    all_contrasts = get_contrasts(exp)
    np.random.seed(1)
    rand_inds = np.random.randint(low=0, high=len(subjects), size=n_subs)
    for subject in subjects[rand_inds]:
        if os.path.exists(f'{HCP_DIR}/func_files/{subject}_tfMRI_{exp}_level2_hp200_s{smoothing}{space}.dscalar.nii'):
            func_map = nib.load(f'{HCP_DIR}/func_files/{subject}_tfMRI_{exp}_level2_hp200_s{smoothing}{space}.dscalar.nii')
            for contrast in contrasts:
                contrast_map = get_contrast_map(func_map.get_fdata(), all_contrasts, contrast)
                if np.max(np.abs(contrast_map)) > 1000000:
                    print(f'subject {subject} bad map')
                    continue
                func_maps[contrast].append(contrast_map)
    return func_maps


def append_extra_hcp_data(df):
    """
    Add HCP demographic and behavioral data to DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with subject IDs
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added HCP columns
    """
    restricted_df = pd.read_csv(RESTRICTED_DATA_PATH) 
    restricted_df['subject'] = [str(sub) for sub in restricted_df['Subject']]

    df = pd.merge(df, restricted_df[['subject', 'Handedness']], on='subject')
    df['handedness_mag'] = np.abs(df['Handedness'])
    df['handedness_dir'] = [1 if h>0 else -1 for h in df['Handedness']]
    df['handedness_categ'] = ['R' if h>40 else 'L' if h<-40 else 'A' for h in df['Handedness']]

    unrestricted_df = pd.read_csv(UNRESTRICTED_DATA_PATH)
    unrestricted_df['subject'] = [str(sub) for sub in unrestricted_df['Subject']]

    df = pd.merge(df, unrestricted_df, on='subject')
    return df


# get average contrasts
def get_average_contrast(exp, contrast, subjects, space='_MSMAll', smoothing=2, out_fn=None):
    """
    Get average contrast map across subjects.
    
    Parameters
    ----------
    exp : str
        Experiment name
    contrast : str
        Contrast to average
    subjects : list
        Subject IDs to include
    space : str, optional
        Space for analysis
    smoothing : int, optional
        Smoothing kernel size
    out_fn : str, optional
        Output filename
        
    Returns
    -------
    array-like
        Average contrast map
    """
    func_maps = []
    for subject in tqdm(subjects):
        if os.path.exists(f'{HCP_DIR}/func_files/{subject}_tfMRI_{exp}_level2_hp200_s{smoothing}{space}.dscalar.nii'):
            func_map = nib.load(f'{HCP_DIR}/func_files/{subject}_tfMRI_{exp}_level2_hp200_s{smoothing}{space}.dscalar.nii')
            contrasts = get_contrasts(exp)
            contrast_map = get_contrast_map(func_map.get_fdata(), contrasts, contrast)
            if np.max(np.abs(contrast_map)) > 1000000:
                print(f'subject {subject} bad map')
                continue
            func_maps.append(contrast_map)
        else:
            continue
            
    mean_map = np.mean(func_maps, 0)
    
    if out_fn is not None:
        np.save(out_fn, mean_map)
        
    return mean_map


# get average contrasts
def get_average_contrasts(exp, contrasts, subjects, space='_MSMAll', smoothing=2, out_fn=None, metric='average', thresh=None):
    """
    Get average maps for multiple contrasts across subjects.
    
    Parameters
    ----------
    exp : str
        Experiment name
    contrasts : list
        Contrasts to average
    subjects : list
        Subject IDs to include
    space : str, optional
        Space for analysis
    smoothing : int, optional
        Smoothing kernel size
    out_fn : str, optional
        Output filename
    metric : str, optional
        Metric for averaging
    thresh : float, optional
        Threshold for data
        
    Returns
    -------
    dict
        Dictionary of average maps per contrast
    """
    func_maps = {contrast:[] for contrast in contrasts}
    all_contrasts = get_contrasts(exp)
    for subject in tqdm(subjects):
        if os.path.exists(f'{HCP_DIR}/func_files/{subject}_tfMRI_{exp}_level2_hp200_s{smoothing}{space}.dscalar.nii'):
            func_map = nib.load(f'{HCP_DIR}/func_files/{subject}_tfMRI_{exp}_level2_hp200_s{smoothing}{space}.dscalar.nii')
            for contrast in contrasts:
                contrast_map = get_contrast_map(func_map.get_fdata(), all_contrasts, contrast)
                if np.max(np.abs(contrast_map)) > 1000000:
                    print(f'subject {subject} bad map')
                    continue
                func_maps[contrast].append(contrast_map)
        else:
            continue
            
    if metric == 'average':
        mean_maps = {contrast: np.mean(func_maps[contrast], 0) for contrast in contrasts}
    elif metric == 'hist':
        assert thresh is not None
        mean_maps = {contrast: np.sum(np.array(func_maps[contrast]) > thresh, 0) for contrast in contrasts}
    else:
        raise NotImplementedError()
    
    if out_fn is not None:
        np.save(out_fn, mean_maps)
        
    return mean_maps

def get_contrast_map(func_maps, contrasts, contrast):
    """
    Extract specific contrast map from functional data.
    
    Parameters
    ----------
    func_maps : array-like
        Functional map data
    contrasts : list
        List of available contrasts
    contrast : str
        Contrast to extract
        
    Returns
    -------
    array-like
        Extracted contrast map
    """
    if 'CUSTOM' in contrast:
        categs = contrast.split(':')[1]
        pos, neg = categs.split('-')
        poss = pos.split('+')
        negs = neg.split('+')
        data = 0
        for pos in poss:
            data = data + (1/len(poss))*func_maps[contrasts.index(pos)]
        for neg in negs:
            data = data - (1/len(negs))*func_maps[contrasts.index(neg)]
    else:
        data = func_maps[contrasts.index(contrast)]        
    return data    

def compute_contrast(sub, exp, contrast, atlas_img, inds,
                     out_type: Literal['parcellated', 'fsaverage', 'raw'],
                     mask_exp=None, mask_contrast=None, mask_thresh=None, mask_reflect=None,
                     func_img=None,
                    ):
    """
    Compute contrast maps for HCP data with optional masking and output formats.
    
    Parameters
    ----------
    sub : str
        Subject ID
    exp : str
        Experiment name (e.g., 'WM', 'LANGUAGE')
    contrast : str
        Contrast name to compute
    atlas_img : array-like
        Atlas image data
    inds : array-like
        Indices of ROIs in atlas
    out_type : {'parcellated', 'fsaverage', 'raw'}
        Output format type
    mask_exp : str, optional
        Experiment to use for masking
    mask_contrast : str, optional
        Contrast to use for masking
    mask_thresh : float, optional
        Threshold for mask
    mask_reflect : str, optional
        Hemisphere to reflect mask from ('lh' or 'rh')
    func_img : array-like, optional
        Pre-loaded functional image data
        
    Returns
    -------
    array-like
        Contrast map in requested output format
    """
    contrasts = get_contrasts(exp)
    if func_img is None:
        func_img = nib.load(f'{HCP_DIR}/func_files/{sub}_tfMRI_{exp}_level2_hp200_s2.dscalar.nii').get_fdata()
    else:
        func_img = copy.copy(func_img)

    if 'CUSTOM' in contrast:
        categs = contrast.split(':')[1]
        pos, neg = categs.split('-')
        poss = pos.split('+')
        negs = neg.split('+')
        func_map = 0
        for pos in poss:
            func_map = func_map + (1/len(poss))*func_img[contrasts.index(pos)]
        for neg in negs:
            func_map = func_map - (1/len(negs))*func_img[contrasts.index(neg)]
    else:
        func_map = func_img[contrasts.index(contrast)]
    
    if mask_contrast is not None:
        mask_contrast = compute_contrast(sub, mask_exp, mask_contrast, atlas_img, inds, func_img=func_img, out_type='raw')
        if mask_reflect is not None:
            # reflect the mask contrast from one hemisphere to the other for symmetric masking
            verts_per_hemi = mask_contrast.shape[0]//2
            if mask_reflect == 'lh':
                mask_contrast[verts_per_hemi:] = mask_contrast[:verts_per_hemi]
            elif mask_reflect == 'rh':
                mask_contrast[:verts_per_hemi] = mask_contrast[verts_per_hemi:]
            else:
                raise ValueError()
        func_map[mask_contrast < mask_thresh] = np.nan
        
    if out_type == 'parcellated':
        sums = []
        for ind in inds:
            roi_dat = func_map[atlas_img == ind]
            sums.append(np.sum(roi_dat[roi_dat>0])/len(roi_dat))
        return np.array(sums)
    elif out_type == 'raw':
        return func_map
    elif out_type == 'fsaverage':
        return hcp_to_fsaverage(func_map)
    else:
        raise ValueError()
    
def clean_contrast_name(contrast_name):
    """
    Clean contrast name by replacing special characters and formatting.
    
    Parameters
    ----------
    contrast_name : str
        Contrast name to clean
        
    Returns
    -------
    str
        Cleaned contrast name
    """
    return str.title(contrast_name.replace('CUSTOM:','').replace('-', u' \u2013 ').replace('FACES', 'EMOTIONAL FACE').replace('SHAPES', 'SHAPE')).replace('Tom', 'TOM')