import numpy as np
from nilearn.surface import load_surf_data
from nilearn.image import index_img, new_img_like
import nibabel as nib
from scipy.io import loadmat
from mne.label import _read_annot
import os
from tqdm import tqdm
import pdb

from .commons import BIDS_DIR
from . import SUBNUMS

BENSON_ROIS = {'V1': 1, 'V2':2, 'V3':3, 'hV4':4, 'VO1':5, 'VO2':6, 'LO1':7, 'LO2':8, 'TO1':9, 'TO2':10, 'V3b':11, 'V3a':12}
APARC_ROIS = {
    'unknown':0,
    'bankssts':1,
    'caudalanteriorcingulate':2,
    'caudalmiddlefrontal':3,
    'corpuscallosum':4,
    'cunear':5,
    'entorhinal':6,
    'fusiform': 7,
    'inferiorparietal':8,
    'inferiortemporal':9,
    'isthmuscingulate':10,
    'lateraloccipital':11,
    'lateralorbitofrontal':12,
    'lingual':13,
    'medialorbitofrontal':14,
    'middletemporal':15,
    'parahippocampal':16,
    'paracentral':17,
    'parsopercularis':18,
    'parsorbitalis':19,
    'parstriangularis':20,
    'pericalcarine':21,
    'postcentral':22,
    'posteriorcingulate':23,
    'precentral':24,
    'precuneus':25,
    'rostralanteriorcingulate':26,
    'rostralmiddlefrontal':27,
    'superiorfrontal':28,
    'superiorparietal':29,
    'superiortemporal':30,
    'supramarginal':31,
    'frontalpole':32,
    'temporalpole':33,
    'transversetemporal':34,
    'insula':35,
}
GLASSER_ROIS_LONG = {'V1':1, 'medial superior temporal area':2, 'V6':3, 'V2':4, 'V3':5, 'V4':6, 'V8':7, 'primary motor cortex':8,
                    'primary sensory cortex':9, 'FEF':10, 'premotor eye field':11, 'area 55b':12, 'area V3A':13,
                    'retrosplenial complex':14, 'parieto-occipital sulcus area 2':15, 'V7':16, 'intraparietal sulcus area 1':17,
                    'fusiform face complex':18, 'V3B':19, 'LO1':20, 'LO2':21, 'posterior inferotemporal complex':22, 'MT':23,
                    'A1':24, 'perisylvian language area':25, 'superior frontal language area':26, 'precuneus visual area':27,
                    'superior temporal visual area':28, 'medial area 7P':29, '7m':30, 'parieto-occipital sulcus area 1':31,
                    '23d':32, 'area ventral 23 a+b':33, 'area dorsal 23 a+b':34, '31p ventral':35, '5m':36, '5m ventral':37,
                    '23c':38, '5L':39, 'dorsal area 24d':40, 'ventral area 24d':41, 'lateral area 7A':42,
                    'supplemetary and cingulate eye field':43, '6m anterior':44, 'medial area 7a':45, 'lateral area 7p':46,
                    '7pc':47, 'lateral intraparietal ventral':48, 'ventral intraparietal complex':49, 'medial intraparietal area':50,
                    'area 1':51, 'area 2':52,'area 3a':53, 'dorsal area 6':54, '6mp':55, 'ventral area 6':56, 'posterior 24 prime':57,
                    '33 prime':58, 'anterior 24 prime':59, 'p32 prime':60, 'a24':61, 'dorsal 32':62, '8BM':63, 'p32':64, '10r':65,
                    '47m':66, '8Av':67, '8Ad':68, 'area 9 middle':69, 'area 8B lateral':70, 'area 9 posterior':71, '10d':72, '8C':73,
                    'area 44':74, 'area 45':75, 'area 47l':76, 'anterior 47r':77, 'rostral area 6':78, 'area IFJa':79, 'area IFJp':80,
                    'area IFSp':81, 'area IFSa':82, 'area posterior 9-46v':83, 'area 46':84, 'area anterior 9-46v':85, 'area 9-46d':86,
                    'area 9 anterior':87, '10v':88, 'anterior 10p':89, 'polar 10p':90, 'area 11l':91, 'area 13l': 92, 'OFC':93, '47s':94,
                    'lateral intraparietal dorsal':95, 'area 6 anterior':96, 'inferior 6-8 transitional area':97,
                    'superior 6-8 transitional area':98, 'area 43':99, 'area OP4/PV':100, 'area OP1/SII':101, 'area OP2-3/VS':102,
                    'area 52':103, 'retroinsular cortex':104, 'area PFcm':105, 'posterior insula area 2':106, 'area TA2':107,
                    'frontal opercular area 4':108, 'middle insular area':109, 'pirform cortex':110, 'anterior ventral insular area':111,
                    'anterior angranular insula complex':112, 'frontal opercular area 1':113, 'frontal opercular area 3':114,
                    'frontal opercular area 2':115, 'area PFt':116, 'anterior intraparietal area':117, 'entorhinal cortex':118,
                    'preSubiculum':119, 'hippocampus':120, 'proStriate area':121, 'perirhinal ectorhinal cortex':122, 'area STGa':123,
                    'parabelt complex':124, 'A5':125, 'parahippocampal area 1':126, 'parahippocampal area 3':127, 'STSd anterior':128,
                    'STSd posterior':129, 'STSv posterior':130, 'TG dorsal':131, 'TE1 anterior':132, 'TE1 posterior':133, 'TE2 anterior':134,
                    'TF':135, 'TE2 posterior':136, 'PHT':137, 'PH':138, 'temporoparietooccipital junction 1':139,
                    'temporoparietooccipital junction 2':140, 'superior 6-8':141, 'dorsal transitional visual area':142, 'PGp':143,
                    'intraparietal 2':144, 'intraparietal 1':145, 'intraparietal 0':146, 'PF opercular':147, 'PF complex':148, 'PFm complex':149,
                    'PGi':150, 'PGs':151, 'V6A':152, 'ventromedial visual area 1':153, 'ventromedial visual area 3':154, 'parahippocampal area 2':155,
                    'V4t':156, 'FST':157, 'V3CD':158, 'lateral occipital 3':159, 'ventromedial visual area 2':160, '31pd':161, '31a':162,
                    'ventral visual complex':163, 'area 25':164, 's32':165, 'posterior OFC complex':166, 'posterior insular 1':167,
                    'insular granular complex':168, 'frontal opercular 5':169, 'posterior 10p':170, 'posterior 47r':171, 'TG ventral':172,
                    'medial belt complex':173, 'lateral belt complex':174, 'A4':175, 'STSv anterior':176, 'TE1 middle':177, 'parainsular area':178,
                    'anterior 32 prime':179, 'posterior 24':180}
GLASSER_ROIS = {'V1':1, 'MST':2, 'V6':3, 'V2':4, 'V3':5, 'V4':6, 'V8':7, '4':8,
                    '3b':9, 'FEF':10, 'PEF':11, '55b':12, 'V3A':13,
                    'RSC':14, 'POS2':15, 'V7':16, 'IPS1':17,
                    'FFC':18, 'V3B':19, 'LO1':20, 'LO2':21, 'PIT':22, 'MT':23,
                    'A1':24, 'PSL':25, 'SFL':26, 'PCV':27,
                    'STV':28, '7Pm':29, '7m':30, 'POS1':31,
                    '23d':32, 'v23ab':33, 'd23ab':34, '31pv':35, '5m':36, '5mv':37,
                    '23c':38, '5L':39, '24dd':40, '24dv':41, '7AL':42,
                    'SCEF':43, '6ma':44, '7Am':45, '7Pl':46,
                    '7PC':47, 'LIPv':48, 'VIP':49, 'MIP':50,
                    '1':51, '2':52,'3a':53, '6d':54, '6mp':55, '6v':56, 'p24pr':57,
                    '33pr':58, 'a24pr':59, 'p32pr':60, 'a24':61, 'd32':62, '8BM':63, 'p32':64, '10r':65,
                    '47m':66, '8Av':67, '8Ad':68, '9m':69, '8BL':70, '9p':71, '10d':72, '8C':73,
                    '44':74, '45':75, '47l':76, 'a47r':77, '6r':78, 'IFJa':79, 'IFJp':80,
                    'IFSp':81, 'IFSa':82, 'p9-46v':83, '46':84, 'a9-46v':85, '9-46d':86,
                    '9a':87, '10v':88, 'a10p':89, '10pp':90, '11l':91, '13l': 92, 'OFC':93, '47s':94,
                    'LIPd':95, '6a':96, 'i6-8':97,
                    's6-8':98, '43':99, 'OP4':100, 'OP1':101, 'OP2-3':102,
                    '52':103, 'RI':104, 'PFcm':105, 'PoI2':106, 'TA2':107,
                    'FOP4':108, 'MI':109, 'Pir':110, 'AVI':111,
                    'AAIC':112, 'FOP1':113, 'FOP3':114,
                    'FOP2':115, 'PFt':116, 'AIP':117, 'EC':118,
                    'PreS':119, 'H':120, 'ProS':121, 'PeEc':122, 'STGa':123,
                    'PBelt':124, 'A5':125, 'PHA1':126, 'PHA3':127, 'STSda':128,
                    'STSdp':129, 'STSvp':130, 'TGd':131, 'TE1a':132, 'TE1p':133, 'TE2a':134,
                    'TF':135, 'TE2p':136, 'PHT':137, 'PH':138, 'TPOJ1':139,
                    'TPOJ2':140, 'TPOJ3':141, 'DVT':142, 'PGp':143,
                    'IP2':144, 'IP1':145, 'IP0':146, 'PFop':147, 'PF':148, 'PFm':149,
                    'PGi':150, 'PGs':151, 'V6A':152, 'VMV1':153, 'VMV3':154, 'PHA2':155,
                    'V4t':156, 'FST':157, 'V3CD':158, 'LO3':159, 'VMV2':160, '31pd':161, '31a':162,
                    'VVC':163, '25':164, 's32':165, 'pOFC':166, 'PoI1':167,
                    'Ig':168, 'FOP5':169, 'p10p':170, 'p47r':171, 'TGv':172,
                    'MBelt':173, 'LBelt':174, 'A4':175, 'STSva':176, 'TE1m':177, 'PI':178,
                    'a32pr':179, 'p24':180}

RAJIMEHR_ROIS = {'Broca': 7, 'SFL': 5, '55b': 6, 'PSL': 3, 'STGa': 4, 'STSa': 8, 'STSp': 1, 'PGi': 2}
RAJIMEHR_ROIS_ORDERED = {'STSp': 1, 'PGi': 2, 'PSL': 3, 'STGa': 4, 'SFL': 5, '55b': 6, 'Broca': 7, 'STSa': 8}             

def _parse_atlas(sub, atlas_name, wm_label_mm=None, wm_merged=False, b0_space=False):
    """
    Load and parse atlas data with configurable white matter handling.
    
    Parameters
    ----------
    sub : str
        Subject ID
    atlas_name : str
        Name of atlas to load
    wm_label_mm : float, optional
        White matter label size in mm
    wm_merged : bool, optional
        Whether white matter is merged
    b0_space : bool, optional
        Whether to load atlas in b0 space
        
    Returns
    -------
    np.ndarray
        Atlas volume data
        
    Raises
    ------
    ValueError
        If incompatible white matter options specified
    """
    if wm_merged:
        assert wm_label_mm is not None
    else:
        if wm_label_mm is not None:
            raise NotImplementedError()
    wm_tag = f'_labelwm-{wm_label_mm}mm' if wm_label_mm is not None else ''
    merged = "_merged" if wm_merged else ""
    space = 'T1w.' if not b0_space else ''
    b0 = "_in_b0" if b0_space else ''
    try:
        atlas = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/{space}{atlas_name}{wm_tag}{merged}{b0}.nii.gz').get_fdata()
    except:
        atlas = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/{space}{atlas_name}{wm_tag}{merged}{b0}.nii').get_fdata()
    return atlas

def _get_aparc_roi_mask(sub, roi_name, hemisphere='both', **kwargs):
    """
    Get binary mask for ROI from aparc+aseg atlas.
    
    Parameters
    ----------
    sub : str
        Subject ID
    roi_name : str
        Name of ROI from APARC_ROIS
    hemisphere : {'both', 'lh', 'rh'}, optional
        Hemisphere to mask, default 'both'
    **kwargs
        Additional arguments passed to _parse_atlas
        
    Returns
    -------
    np.ndarray
        Binary mask array
        
    Raises
    ------
    ValueError
        If invalid hemisphere specified
    """
    if hemisphere not in ['lh','rh','both']:
        raise ValueError()
    roi_ind = APARC_ROIS[roi_name]
    atlas = _parse_atlas(sub, 'aparc+aseg', **kwargs)
    if hemisphere == 'both':
        mask = np.logical_or(atlas == 1000+roi_ind,atlas == 2000+roi_ind)
    elif hemisphere == 'lh':
        mask = (atlas == 1000+roi_ind)
    elif hemisphere == 'rh':
        mask = atlas == 2000+roi_ind
    return mask


def _get_retinotopic_roi_mask(sub, roi_name, hemisphere='both', **kwargs):
    """
    Get binary mask for ROI from Benson 2014 retinotopic atlas.
    
    Parameters
    ----------
    sub : str
        Subject ID
    roi_name : str
        Name of ROI from BENSON_ROIS
    hemisphere : {'both', 'lh', 'rh'}, optional
        Hemisphere to mask, default 'both'
    **kwargs
        Additional arguments passed to _parse_atlas
        
    Returns
    -------
    np.ndarray
        Binary mask array
        
    Raises
    ------
    ValueError
        If invalid hemisphere specified
    """
    if hemisphere not in ['lh', 'rh', 'both']:
        raise ValueError()
    roi_ind = BENSON_ROIS[roi_name]
    mask = None
    atlas = _parse_atlas(sub, 'benson14_varea', **kwargs)
    mask = (atlas == roi_ind)
    if hemisphere != 'both':
        cortex_mask = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/T1w.{hemisphere}.cortex.nii').get_fdata()
    else:
        cortex_mask = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/T1w.cortex.nii').get_fdata()
    mask = np.logical_and(mask, cortex_mask > .5)
    return mask


def _get_glasser_roi_mask(sub, roi_name, hemisphere='both', **kwargs):
    """
    Get binary mask for ROI from Glasser Multi-Modal Parcellation.
    
    Parameters
    ----------
    sub : str
        Subject ID
    roi_name : str
        Name of ROI from GLASSER_ROIS
    hemisphere : {'both', 'lh', 'rh'}, optional
        Hemisphere to mask, default 'both'
    **kwargs
        Additional arguments passed to _parse_atlas
        
    Returns
    -------
    np.ndarray
        Binary mask array
        
    Raises
    ------
    ValueError
        If invalid hemisphere specified
    """
    if hemisphere not in ['lh','rh','both']:
        raise ValueError()
    roi_ind = GLASSER_ROIS[roi_name]
    mask = None
    atlas = _parse_atlas(sub, 'HCPMMP1+aseg', **kwargs)
    if hemisphere == 'both':
        mask = np.logical_or(atlas == 1000+roi_ind, atlas == 2000+roi_ind)
    elif hemisphere == 'lh':
        mask = (atlas == 1000+roi_ind)
    elif hemisphere == 'rh':
        mask = atlas == 2000+roi_ind
    return mask     
    

def get_single_roi_mask(sub, roi_name, hemisphere, **kwargs):
    """
    Get binary mask for a single ROI.
    
    Parameters
    ----------
    sub : str
        Subject ID
    roi_name : str
        Name of ROI to mask
    hemisphere : {'both', 'lh', 'rh'}
        Hemisphere to mask
    **kwargs
        Additional arguments passed to atlas parsing functions
        
    Returns
    -------
    np.ndarray
        Binary mask array
        
    Raises
    ------
    ValueError
        If roi_name not found in any atlas
    """
    if roi_name in GLASSER_ROIS.keys():
        mask = _get_glasser_roi_mask(sub, roi_name, hemisphere, **kwargs)
    elif roi_name in BENSON_ROIS.keys():
        mask = _get_retinotopic_roi_mask(sub, roi_name, hemisphere, **kwargs)
    elif roi_name in APARC_ROIS.keys():
        mask = _get_aparc_roi_mask(sub, roi_name, hemisphere, **kwargs)
    elif roi_name == 'cortex':
        if hemisphere != 'both':
            mask = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/T1w.{hemisphere}.cortex.nii').get_fdata()
        else:
            mask = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/T1w.cortex.nii').get_fdata()
    else:
        raise NotImplementedError(f'{roi_name} not found')
    return mask
        

def get_roi_mask(sub, rois_to_include, rois_to_exclude, hemisphere, **kwargs):
    """
    Get combined binary mask for multiple ROIs.
    
    Creates a mask including specified ROIs and optionally excluding others.
    
    Parameters
    ----------
    sub : str
        Subject ID
    rois_to_include : list
        ROI names to include in mask
    rois_to_exclude : list or None
        ROI names to exclude from mask
    hemisphere : {'both', 'lh', 'rh'}
        Hemisphere to mask
    **kwargs
        Additional arguments passed to get_single_roi_mask
        
    Returns
    -------
    np.ndarray
        Binary mask array
        
    Raises
    ------
    ValueError
        If no ROIs specified to include
    """
    if rois_to_include is None or len(rois_to_include) == 0:
        raise ValueError('requires at least one ROI to include')
        
    inclusion = []
    for roi in rois_to_include:
        inclusion.append(get_single_roi_mask(sub, roi, hemisphere, **kwargs))
    inclusion = np.logical_or.reduce([mask for mask in inclusion])
    if rois_to_exclude is None or len(rois_to_exclude) == 0:
        pass
    else:
        exclusion = []
        for roi in rois_to_exclude:
            exclusion.append(get_single_roi_mask(sub, roi, hemisphere, **kwargs))
        exclusion = np.logical_or.reduce([mask for mask in exclusion])
        inclusion[exclusion] = 0
    return inclusion


def extract_AP_segment(bin_mask, n_segments, segment):
    """
    Extract anterior-posterior segment from binary mask.
    
    Divides mask into n_segments along the anterior-posterior axis and
    returns the specified segment.
    
    Parameters
    ----------
    bin_mask : np.ndarray
        Binary mask to segment
    n_segments : int
        Number of segments to divide into
    segment : int
        Zero-based index of segment to extract
        
    Returns
    -------
    np.ndarray
        Binary mask containing only the specified segment
    """
    y_loc = np.nonzero(bin_mask)[1]
    bounds = np.linspace(np.min(y_loc), np.max(y_loc)+1, num=n_segments+1)
    inds = np.nonzero(np.logical_and(y_loc >= bounds[segment], y_loc < bounds[segment+1]))
    new_mask = np.zeros_like(bin_mask)
    new_mask[tuple([dim_inds[inds] for dim_inds in np.nonzero(bin_mask)])] = 1
    return new_mask


def get_func_mask(sub, func_mask_name, mask_sm_fwhm=4, func_thresh=3, return_raw=False):
    """
    Get functional activation mask.
    
    Parameters
    ----------
    sub : str
        Subject ID
    func_mask_name : str
        Name of functional contrast
    mask_sm_fwhm : float, optional
        Smoothing kernel FWHM in mm, default 4
    func_thresh : float, optional
        Threshold for mask in -log10(p), default 3 (p < 0.001)
    return_raw : bool, optional
        Whether to return raw values instead of thresholded mask
        
    Returns
    -------
    np.ndarray
        Binary mask array or raw values if return_raw=True
    """
    sm_tag = f'_sm-{mask_sm_fwhm}mm' if mask_sm_fwhm else ''
    func_mask_img = nib.load(f'{BIDS_DIR}/derivatives/matlab/spm/sub-{sub:02d}/SPM-floc_vol-T1w{sm_tag}/{func_mask_name}_log10p.nii')
    if return_raw:
        assert func_thresh is None
        func_mask = func_mask_img.get_fdata()
    else:
        func_mask = func_mask_img.get_fdata()>func_thresh
    return func_mask


def get_full_mask(sub, rois_to_include, rois_to_exclude, hemisphere, 
                  func_mask='vis-on-vs-off', mask_sm_fwhm=4, func_thresh=3,
                  AP_segments=None, AP_segment=None,
                  **kwargs):
    """
    Get combined anatomical and functional mask.
    
    Creates a mask combining ROIs and functional activation, with optional
    anterior-posterior segmentation.
    
    Parameters
    ----------
    sub : str
        Subject ID
    rois_to_include : list
        ROI names to include in mask
    rois_to_exclude : list or None
        ROI names to exclude from mask
    hemisphere : {'both', 'lh', 'rh'}
        Hemisphere to mask
    func_mask : str or None, optional
        Name of functional contrast for masking
    mask_sm_fwhm : float, optional
        Smoothing kernel FWHM in mm for functional mask
    func_thresh : float, optional
        Threshold for functional mask in -log10(p)
    AP_segments : int, optional
        Number of anterior-posterior segments
    AP_segment : int, optional
        Zero-based index of AP segment to extract
    **kwargs
        Additional arguments passed to get_roi_mask
        
    Returns
    -------
    np.ndarray
        Binary mask array
    """
    roi_mask = get_roi_mask(sub, rois_to_include, rois_to_exclude, hemisphere, **kwargs)
    if AP_segments is not None:
        assert AP_segment is not None
        roi_mask = extract_AP_segment(roi_mask, AP_segments, AP_segment)
    if func_mask is None or func_thresh is None:
        full_mask = roi_mask
    else:
        functional_mask = get_func_mask(sub, func_mask, mask_sm_fwhm=mask_sm_fwhm, func_thresh=func_thresh)
        full_mask = np.logical_and(roi_mask, functional_mask)
    return full_mask


def map_parcellated_values(lh_dict, rh_dict, sub, annot_name='glasser', map_type='vol'):
    """
    Map ROI values to volume or surface space.
    
    Parameters
    ----------
    lh_dict : dict
        Left hemisphere ROI name:value mapping
    rh_dict : dict
        Right hemisphere ROI name:value mapping
    sub : str
        Subject ID
    annot_name : {'glasser', 'aparc'}, optional
        Atlas to use for mapping
    map_type : {'vol', 'surf'}, optional
        Whether to map to volume or surface space
        
    Returns
    -------
    nibabel.Nifti1Image or np.ndarray
        Mapped values in requested space
        
    Raises
    ------
    NotImplementedError
        If unsupported atlas or map type specified
    """
    if map_type not in ['vol', 'surf']:
        raise ValueError('type must be vol or surf')

    if annot_name.lower() == 'glasser':
        annot_fn = 'T1w.HCPMMP1+aseg' if map_type == 'vol' else 'HCPMMP1'
        parcellation = GLASSER_ROIS        
    elif annot_name.lower() == 'aparc':
        annot_fn = 'T1w.aparc+aseg' if map_type == 'vol' else 'aparc'
        parcellation = APARC_ROIS
    else:
        raise NotImplementedError()

    if map_type == 'vol':
        try:
            atlas = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/{sub}/mri/{annot_fn}.nii.gz')
        except:
            atlas = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/{sub}/mri/{annot_fn}.nii')
        atlas_dat = atlas.get_fdata()
        parcelled_map = np.full_like(atlas_dat, np.nan)
        for hemi, values_dict in {'lh': lh_dict, 'rh': rh_dict}.items():
            base_val = 1000 if hemi == 'lh' else 2000
            for key, val in values_dict.items():
                vox_mask = atlas_dat == base_val+parcellation[key]
                parcelled_map[vox_mask] = val
        parcelled_vol = new_img_like(atlas, parcelled_map)
        return parcelled_vol 
    elif map_type == 'surf':
        lh_annot = load_surf_data(f'{BIDS_DIR}/derivatives/freesurfer/{sub}/label/lh.{annot_fn}.annot') 
        rh_annot = load_surf_data(f'{BIDS_DIR}/derivatives/freesurfer/{sub}/label/rh.{annot_fn}.annot') 
        annot = np.concatenate((1000+lh_annot, 2000+rh_annot))
        parcelled_map = np.full_like(annot, np.nan, dtype=float)
        for hemi, values_dict in {'lh': lh_dict, 'rh': rh_dict}.items():
            base_val = 1000 if hemi == 'lh' else 2000
            for key, val in values_dict.items():
                mask = annot == base_val+parcellation[key]
                parcelled_map[mask] = val
        return parcelled_map 


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
    surf_map = np.zeros_like(surf_annot, dtype=float)
    for ii, ind in enumerate(atlas_inds):
        surf_map[surf_annot == ind] = atlas_results[ii]
    return surf_map


def get_annot(atlas):
    """
    Load and concatenate left and right hemisphere annotations.
    
    Parameters
    ----------
    atlas : str
        Name of atlas annotation to load
        
    Returns
    -------
    np.ndarray
        Concatenated left and right hemisphere annotations with 
        hemisphere offsets applied
    """
    lh_annot = load_surf_data(f'{BIDS_DIR}/derivatives/freesurfer/fsaverage/label/lh.{atlas}.annot') 
    rh_annot = load_surf_data(f'{BIDS_DIR}/derivatives/freesurfer/fsaverage/label/rh.{atlas}.annot') 
    annot = np.concatenate((1000+lh_annot, 2000+rh_annot))
    return annot


def find_overlapping_rois(atlas, mask, nvox_thresh=0):
    """
    Find ROIs that overlap with a mask above a threshold.
    
    Parameters
    ----------
    atlas : array-like
        Atlas annotation array
    mask : array-like
        Binary mask array
    nvox_thresh : int, optional
        Minimum number of overlapping voxels required, default 0
        
    Returns
    -------
    list
        List of ROI indices that overlap with mask above threshold
    """
    bad_rois = []
    for roi in np.unique(atlas):
        if roi == 0 or roi > 2000:
            # just look at lh values homotopically
            continue
        if np.sum(np.logical_and(atlas==roi, mask)) + np.sum(np.logical_and(atlas==roi+1000, mask)) > nvox_thresh:
            bad_rois.append(roi)
            bad_rois.append(roi+1000)
    return bad_rois


def find_overlap_of_rois(atlas, mask):
    """
    Calculate overlap between ROIs and a mask.
    
    Parameters
    ----------
    atlas : array-like
        Atlas annotation array
    mask : array-like
        Binary mask array
        
    Returns
    -------
    dict
        Dictionary mapping ROI indices to number of overlapping voxels
    """
    overlap = {}
    for roi in np.unique(atlas.astype(int)):
        if roi == 0 or roi > 2000:
            # just look at lh values homotopically
            continue
        overlap[str(roi)] = np.sum(np.logical_and(atlas==roi, mask)) + np.sum(np.logical_and(atlas==roi+1000, mask))
        overlap[str(roi+1000)] = np.sum(np.logical_and(atlas==roi, mask)) + np.sum(np.logical_and(atlas==roi+1000, mask))
    return overlap


def find_overlapping_rois_group(atlas, nvox_thresh=0):
    """
    Find ROIs that overlap with VTC mask across subject group.
    
    Parameters
    ----------
    atlas : str
        Name of atlas to use
    nvox_thresh : int, optional
        Minimum mean number of overlapping voxels required, default 0
        
    Returns
    -------
    list
        List of ROI indices that overlap with VTC above threshold
        across subjects
    """
    all_overlap = {}
    for sub in tqdm(SUBNUMS):
        try:
            atlas_img = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/T1w.{atlas}+aseg.nii.gz').get_fdata()
        except:
            atlas_img = nib.load(f'{BIDS_DIR}/derivatives/freesurfer/sub-{sub:02d}/mri/T1w.{atlas}+aseg.nii').get_fdata()
        overlap = find_overlap_of_rois(atlas_img, get_full_mask(sub, ['fusiform', 'inferiortemporal'], [], 'both', func_mask=None))
        for roi in overlap.keys():
            if roi in all_overlap.keys():
                all_overlap[roi].append(overlap[roi])
            else:
                all_overlap[roi] = [overlap[roi]]
    all_overlap = {roi: np.mean(overlap) for roi, overlap in all_overlap.items()}
    exclude_rois = [int(roi) for roi in all_overlap if all_overlap[roi] > nvox_thresh]
        
    return exclude_rois