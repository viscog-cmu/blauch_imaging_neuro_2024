from func.hcp import get_atlas_selectivity, EXP_CONTRASTS
import statsmodels.api as sm
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import numpy as np
from nilearn.surface import load_surf_data
from . import BIDS_DIR
import hcp_utils
import copy
import cortex
from func.pycortex_utils import masked_vertex_nans
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
from func.masking import GLASSER_ROIS, RAJIMEHR_ROIS
from statsmodels.stats.multitest import fdrcorrection
from typing import Literal


lh_annot = load_surf_data(f'{BIDS_DIR}/derivatives/freesurfer/fsaverage/label/lh.HCPMMP1.annot') 
rh_annot = load_surf_data(f'{BIDS_DIR}/derivatives/freesurfer/fsaverage/label/rh.HCPMMP1.annot') 
annot = np.concatenate((1000+lh_annot, 2000+rh_annot))

# align indices
hcp_annot = copy.copy(annot)
hcp_annot[hcp_annot<1000] = 0
hcp_annot[np.logical_and(hcp_annot>1000,hcp_annot<2000)] -= 1000
hcp_annot[hcp_annot>2000] -= (2000 - 180)

atlas_img = hcp_utils.mmp['map_all']
HCP_INDS = np.unique(atlas_img) 
HCP_INDS = HCP_INDS[HCP_INDS<361]

# color map for rajimehr
colors = {
    'Broca': 'violet',
    'PSL': 'lime',
    '55b': 'm',
    'SFL': 'purple',
    'STGa': 'gold',
    'STSa': 'darkorange',
    'STSp': 'red',
    'PGi': 'cornflowerblue',
}
color_list = []
for ii, roi in enumerate(RAJIMEHR_ROIS.keys()):
    color_list.append(colors[roi])

categorical_cmap = ListedColormap(color_list, name='rajimehr_novtc')
cm.register_cmap(name='rajimehr_novtc', cmap=categorical_cmap, override_builtin=True)


def atlas_to_surf_map(atlas_results, atlas_inds, surf_annot):
    """
    Map atlas results to surface annotation space.
    
    Parameters
    ----------
    atlas_results : array-like
        Results values for each atlas index
    atlas_inds : array-like
        Atlas indices corresponding to results
    surf_annot : array-like
        Surface annotation array to map results onto
        
    Returns
    -------
    np.ndarray
        Results mapped onto surface annotation space
    """
    surf_map = np.zeros_like(surf_annot, dtype=float)*np.nan
    for ii, ind in enumerate(atlas_inds):
        surf_map[surf_annot == ind] = atlas_results[ii]
    return surf_map


def get_sums(hand_thresh=40, atlas='HCPMMP1', experiments=['WM', 'LANGUAGE', 'SOCIAL', 'EMOTION'], 
             handednesses=['L', 'R', 'B'],
             **kwargs):
    """
    Get task selectivity sums for ROIs across subjects and experiments.
    
    Parameters
    ----------
    hand_thresh : int, optional
        Handedness score threshold, default 40
    atlas : str, optional
        Atlas to use, default 'HCPMMP1'
    experiments : list, optional
        List of experiments to analyze
    handednesses : list, optional
        List of handedness groups to include ('L', 'R', 'B')
    **kwargs
        Additional arguments passed to get_atlas_selectivity
        
    Returns
    -------
    tuple
        (Dictionary of selectivity sums per handedness/experiment/contrast,
         ROI indices)
    """
    all_sums = {key: {} for key in handednesses}
    for exp in experiments: #, 'RELATIONAL', 'MOTOR']: #, 'SOCIAL', 'LANGUAGE']:
        for hand in handednesses:
            if hand == 'B':
                these_sums, inds, subs = get_atlas_selectivity(exp, EXP_CONTRASTS[exp], 
                                                               atlas=atlas, handedness=None, overwrite=False, 
                                                               **kwargs,
                                                              )
            else:
                these_sums, inds, subs = get_atlas_selectivity(exp, EXP_CONTRASTS[exp], 
                                                               atlas=atlas, handedness=hand, hand_thresh=hand_thresh, 
                                                               overwrite=False, 
                                                               **kwargs,
                                                              )
            all_sums[hand][exp] = {}
            for key in these_sums.keys():
                all_sums[hand][exp][key] = pd.DataFrame(data=these_sums[key], index=subs, columns=inds)
    return all_sums, inds


def get_roi_from_atlas(all_sums, contrast, task, handedness, rois, hemi, 
                      atlas: Literal['HCPMMP1', 'rajimehr']='HCPMMP1', subjects=None):
    """
    Extract ROI values from atlas results.
    
    Parameters
    ----------
    all_sums : dict
        Dictionary of activation sums from get_sums()
    contrast : str
        Contrast name
    task : str
        Task name
    handedness : str
        Handedness group
    rois : str or list
        ROI name(s) to extract
    hemi : {'lh', 'rh', 'laterality'}
        Hemisphere or laterality measure
    atlas : {'HCPMMP1', 'rajimehr'}, optional
        Atlas to use, default 'HCPMMP1'
    subjects : list, optional
        Subset of subjects to include
        
    Returns
    -------
    tuple
        (ROI values array, subject IDs)
    """
    subs = all_sums[handedness][task][contrast].index
    if subjects is not None:
        subs = [subject for subject in subjects if subject in subs]
    roi_map = all_sums[handedness][task][contrast].loc[subs].values
    roi_map[roi_map > 100000] = np.nan
    if atlas == 'HCPMMP1':
        all_rois = GLASSER_ROIS
    elif atlas == 'rajimehr':
        all_rois = RAJIMEHR_ROIS
    roi_vals, _ = _roi_from_atlas(roi_map, rois, all_rois, hemi, hemi == 'laterality')
    return roi_vals, subs
        

def _roi_from_atlas(roi_map, rois, all_rois, hemi, target_laterality):
    """
    Helper function to extract ROI values from atlas map.
    
    Parameters
    ----------
    roi_map : array-like
        Map of ROI values
    rois : str or list
        ROI name(s) to extract
    all_rois : dict
        Dictionary mapping ROI names to indices
    hemi : {'lh', 'rh', 'laterality', 'laterality_diff', 'laterality_num_denom'}
        Hemisphere or laterality measure
    target_laterality : bool
        Whether computing laterality measure
        
    Returns
    -------
    tuple
        (ROI values array, ROI indices)
    """
    if hemi in ['laterality', 'laterality_diff', 'laterality_num_denom']:
        if type(rois) is list:
            lh_vals = np.zeros(roi_map.shape[0])
            rh_vals = np.zeros(roi_map.shape[0])
            seed_inds = []
            for seed_roi in rois:
                lh_vals += roi_map[:,all_rois[seed_roi]-1]
                rh_vals += roi_map[:,all_rois[seed_roi]+len(all_rois)-1]
                seed_inds.append(all_rois[seed_roi]-1)
                if not target_laterality:
                    seed_inds.append(all_rois[seed_roi]+len(all_rois)-1)
        else:
            lh_vals = roi_map[:,all_rois[rois]-1]
            rh_vals = roi_map[:,all_rois[rois]+180-1]
            seed_inds = [all_rois[rois]-1]
        seed_vals = (lh_vals - rh_vals)
        if hemi == 'laterality':
            seed_vals = seed_vals/(rh_vals+lh_vals)
        elif hemi == 'laterality_num_denom':
            num = lh_vals-rh_vals
            denom = lh_vals+rh_vals
            # regress out denom from num and return residuals
            mask = ~np.isnan(denom)
            num_nonan = sm.add_constant(denom[mask])
            model = sm.OLS(num[mask], denom[mask])
            results = model.fit()
            seed_vals = np.full_like(denom, np.nan)
            seed_vals[mask] = results.resid
    else:
        add = 0 if hemi.lower() == 'lh' else len(all_rois)
        seed_inds = []
        if type(rois) is list:
            seed_vals = np.zeros(roi_map.shape[0])
            for seed_roi in rois:
                seed_vals += roi_map[:,all_rois[seed_roi]+add-1]
                seed_inds.append(all_rois[seed_roi]-1)
        else:
            seed_vals = roi_map[:,all_rois[rois]+add-1]
            seed_inds = [all_rois[rois]-1]
    return seed_vals, seed_inds    


def spearman_p_to_r(p_value, n):
    """
    Convert Spearman correlation p-value to correlation coefficient.
    
    Parameters
    ----------
    p_value : float
        P-value from Spearman correlation
    n : int
        Sample size
        
    Returns
    -------
    float
        Estimated Spearman correlation coefficient
    """
    import scipy.stats as stats
    # Degrees of freedom
    df = n - 2

    # Inverse of the t-distribution CDF to find the t-value
    t_value = stats.t.ppf(1 - p_value / 2, df)
    
    # Convert t-value to correlation coefficient
    r_value = t_value / np.sqrt(t_value**2 + df)
    
    return r_value


def plot_pairwise_corrs_all_rois(all_sums, task, contrast, target_task, target_contrast, 
                                laterality=True, handedness='R',
                                plot_logp=True, plot_r=True, subtract_mean=False,
                                do_fdr=True, base_fn=None, folder=None,
                                r_max=0.35, logp_max=10, height=512,
                                subjects=None):
    """
    Plot correlation maps between task contrasts across ROIs.
    
    Parameters
    ----------
    all_sums : dict
        Dictionary of activation sums from get_sums()
    task : str
        First task name
    contrast : str
        First contrast name
    target_task : str
        Second task name
    target_contrast : str
        Second contrast name
    laterality : bool, optional
        Whether to compute laterality correlations, default True
    handedness : str, optional
        Handedness group to analyze, default 'R'
    plot_logp : bool, optional
        Whether to plot -log10(p) values, default True
    plot_r : bool, optional
        Whether to plot correlation coefficients, default True
    subtract_mean : bool, optional
        Whether to subtract subject means, default False
    do_fdr : bool, optional
        Whether to apply FDR correction, default True
    base_fn : str, optional
        Base filename for saving plots
    folder : str, optional
        Output folder for saving plots
    r_max : float, optional
        Maximum correlation for colormap
    logp_max : float, optional
        Maximum -log10(p) for colormap
    height : int, optional
        Plot height in pixels
    subjects : list, optional
        Subset of subjects to include
        
    Returns
    -------
    list
        List of plot outputs (masked vertex data)
    """
    if subjects is None:
        subjects = all_sums[handedness][task][contrast].index
    else:
        subjects = [subject for subject in subjects if subject in all_sums[handedness][task][contrast].index]

    roi_map = all_sums[handedness][task][contrast].loc[subjects].values.copy()
    target_map = all_sums[handedness][target_task][target_contrast].loc[subjects].values.copy()
    targ_str = f' target-{target_task}-{target_contrast}'
        
        # get rid of crazy subjects
    good_subs = np.logical_and(roi_map.sum(1) < 10**25,target_map.sum(1) < 10**25)

    # convert to laterality
    if laterality:
        target_map = (target_map[good_subs,:180]- target_map[good_subs,180:])/(target_map[good_subs,180:] + target_map[good_subs,:180])
        roi_map = (roi_map[good_subs,:180]- roi_map[good_subs,180:])/(roi_map[good_subs,180:] + roi_map[good_subs,:180])
        use_inds = np.arange(1,181)
    else:
        use_inds = np.arange(1,361)

    # remove subject-level means
    if subtract_mean:
        roi_map -= roi_map.mean(1).reshape(-1,1)
        target_map -= target_map.mean(1).reshape(-1,1)
        
    rs, ps = [], []
    for roi in range(target_map.shape[1]):
        r, p = spearmanr(target_map[:,roi], roi_map[:,roi], nan_policy='omit')
        rs.append(r)
        ps.append(p)
    rs, ps = np.array(rs), np.array(ps)
    rejected, fdr_ps = fdrcorrection(ps, alpha=0.05)
    
    lat_tag = '_laterality' if laterality else ''

    outputs = []
    if plot_logp:
        stat = -np.sign(rs)*np.log10(ps)
        stat_map = atlas_to_surf_map(stat, use_inds, hcp_annot)
        surf_data = cortex.Vertex(stat_map, subject='fsaverage', cmap='RdBu_r', vmin=-logp_max, vmax=logp_max)
        surf_data.to_json()
        cortex.quickshow(surf_data, with_labels=False, with_rois=False, height=height, cutout='LH' if laterality else None, with_curvature=True)
        plt.title(f'{task}-{contrast}{targ_str}\n -log(p) of pairwise correlations of ' + ('laterality' if laterality else 'selectivity'))
        if base_fn is not None and folder is not None:
            plt.savefig(f'{folder}/{base_fn}{lat_tag}_logp.png', dpi=200)
        plt.show()
        outputs.append(masked_vertex_nans('fsaverage', surf_data))

    if plot_r:
        stat = rs
        if do_fdr:
            stat[np.logical_not(rejected)] = np.nan
        stat_map = atlas_to_surf_map(stat, use_inds, hcp_annot)
        surf_data = cortex.Vertex(stat_map, subject='fsaverage', cmap='RdBu_r', vmin=-r_max, vmax=r_max)
        surf_data.to_json()
        cortex.quickshow(surf_data, with_labels=False, with_rois=False, height=height, cutout='LH' if laterality else None, with_curvature=True)
        plt.title(f'{task}-{contrast}{targ_str}\n parcel pair-wise r of ' + ('laterality' if laterality else 'selectivity'))
        if base_fn is not None and folder is not None:
            fdr_tag = '_fdr' if do_fdr else ''
            plt.savefig(f'{folder}/{base_fn}{lat_tag}_r{fdr_tag}.png', dpi=200)
        plt.show()
        outputs.append(masked_vertex_nans('fsaverage', surf_data))
        
    return outputs