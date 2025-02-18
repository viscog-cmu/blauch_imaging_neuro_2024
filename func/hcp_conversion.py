from neuromaps import transforms as neuro_transforms
import numpy as np
import hcp_utils
import nibabel as nib

def prep_neuromaps_array(grayordinates, is_fsaverage=False, symmetric=True):
    """
    Prepare data arrays for neuromaps transformations.
    
    Parameters
    ----------
    grayordinates : array-like
        Input data in grayordinates or fsaverage space
    is_fsaverage : bool, optional
        Whether input is in fsaverage space, default False
    symmetric : bool, optional
        Whether to use same data for both hemispheres, default True
        
    Returns
    -------
    tuple
        (left hemisphere GIFTI, right hemisphere GIFTI)
    """
    if not is_fsaverage:
        cortex = hcp_utils.cortex_data(grayordinates)
    else:
        cortex = grayordinates
    gifti_l = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(cortex[:cortex.shape[0]//2].astype(np.float32))])
    if symmetric:
        gifti_r = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(cortex[:cortex.shape[0]//2].astype(np.float32))])
    else:
        gifti_r = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(cortex[cortex.shape[0]//2:].astype(np.float32))])
    return (gifti_l, gifti_r)

def neuromaps_to_array(giftis):
    """
    Convert from neuromaps GIFTI format to numpy array.
    
    Parameters
    ----------
    giftis : tuple
        (left hemisphere GIFTI, right hemisphere GIFTI)
        
    Returns
    -------
    np.ndarray
        Combined hemisphere data array
    """
    return np.concatenate((giftis[0].darrays[0].data, giftis[1].darrays[0].data),0)

def hcp_to_fsaverage(data, target_density='164k', is_grayordinates=False, symmetric=True, **kwargs):
    """
    Transform HCP FSLR data to fsaverage space.
    
    Parameters
    ----------
    data : array-like
        Input data in HCP space
    target_density : str, optional
        Target mesh density ('164k' or '32k'), default '164k'
    is_grayordinates : bool, optional
        Whether input is in grayordinates format, default False
    symmetric : bool, optional
        Whether to use symmetric mapping, default True
    **kwargs
        Additional arguments passed to fslr_to_fsaverage
        
    Returns
    -------
    np.ndarray
        Data transformed to fsaverage space
    """
    return neuromaps_to_array(neuro_transforms.fslr_to_fsaverage(prep_neuromaps_array(data, is_grayordinates, symmetric), target_density, **kwargs))

def fsaverage_to_hcp(data, target_density='32k', as_cifti=False, symmetric=True, **kwargs):
    """
    Transform fsaverage data to HCP FSLR space.
    
    Parameters
    ----------
    data : array-like
        Input data in fsaverage space
    target_density : str, optional
        Target mesh density ('32k' or '164k'), default '32k'
    as_cifti : bool, optional
        Whether to return data in CIFTI format, default False
    symmetric : bool, optional
        Whether to use symmetric mapping, default True
    **kwargs
        Additional arguments passed to fsaverage_to_fslr
        
    Returns
    -------
    np.ndarray
        Data transformed to HCP space, optionally in CIFTI format
    """
    fslr = neuro_transforms.fsaverage_to_fslr(prep_neuromaps_array(data, is_fsaverage=True, symmetric=symmetric), target_density, **kwargs)
    if as_cifti:
        return fslr_fill_cifti(fslr)
    else:
        return neuromaps_to_array(fslr)

def fslr_fill_cifti(giftis):
    """
    Convert GIFTI hemisphere pair to CIFTI format.
    
    Parameters
    ----------
    giftis : tuple
        (left hemisphere GIFTI, right hemisphere GIFTI)
        
    Returns
    -------
    np.ndarray
        Data in CIFTI format with subcortical structures filled with zeros
    """
    l = giftis[0].darrays[0].data
    r = giftis[1].darrays[0].data
    func_map = np.zeros(91282,)
    func_map[hcp_utils.struct.cortex_left] = l[hcp_utils.vertex_info.grayl]
    func_map[hcp_utils.struct.cortex_right] = r[hcp_utils.vertex_info.grayr]
    return func_map