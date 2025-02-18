import numpy as np
import os
import nibabel as nib
import cortex 
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio

from .commons import BIDS_DIR
from . import SUBNUMS, FIGS_DIR, SUBNUMS

params_lateral_flatmap_ventral = {
    'figsize': [9, 12],
    'panels': [
    {
        'extent': [0.000, 0.08, 0.48, 0.333],
        'view': {
            'hemisphere': 'left',
            'angle': 'bottom_pivot',
            'surface': 'inflated'
        }
    }, 
    {
        'extent': [0.000, 0.666, 0.48, 0.333],
        'view': {
            'hemisphere': 'left',
            'angle': 'lateral_pivot',
            'surface': 'inflated'
        }
    }, 
        
    {
        'extent': [0.5, 0.08, 0.48, 0.333],
        'view': {
            'hemisphere': 'right',
            'angle': 'bottom_pivot',
            'surface': 'inflated'
        }
    }, 
    {
        'extent': [0.5, 0.666, 0.48, 0.333],
        'view': {
            'hemisphere': 'right',
            'angle': 'lateral_pivot',
            'surface': 'inflated'
        }
    }, 
    {
        'extent': [0.000, 0.36, 1, 0.333],
        'view': {
            'angle': 'flatmap',
            'surface': 'flatmap',
            # 'zoom': [0.250, 0.000, 0.500, 1.000]
        }
    },         
    ]
}

params_lateral_flatmap = {
    'figsize': [9, 9],
    'panels': [
    {
        'extent': [0.02, 0.5, 0.44, 0.44],
        'view': {
            'hemisphere': 'left',
            'angle': 'lateral_pivot',
            'surface': 'inflated'
        }
    }, 
    {
        'extent': [0.52, 0.5, 0.44, 0.44],
        'view': {
            'hemisphere': 'right',
            'angle': 'lateral_pivot',
            'surface': 'inflated'
        }
    }, 
    {
        'extent': [0, 0.12, 0.96, 0.48],
        'view': {
            'angle': 'flatmap',
            'surface': 'flatmap',
            # 'zoom': [0.250, 0.000, 0.500, 1.000]
        }
    },         
    ]
}

params_lateral = {
    'figsize': [9, 4.5],
    'panels': [
        {
        'extent': [0.000, 0, 0.48, 1],
        'view': {
            'hemisphere': 'left',
            'angle': 'lateral_pivot',
            'surface': 'inflated'
        }
    }, 
    {
        'extent': [0.5, 0, 0.48, 1],
        'view': {
            'hemisphere': 'right',
            'angle': 'lateral_pivot',
            'surface': 'inflated'
        }
    },
    ],
}

params_lateral_ventral = {
    'figsize': [9, 9],
    'panels': [
    {
        'extent': [0.000, 0.05, 0.48, 0.48],
        'view': {
            'hemisphere': 'left',
            'angle': 'bottom_pivot',
            'surface': 'inflated'
        }
    }, 
    {
        'extent': [0.000, 0.4, 0.48, 0.48],
        'view': {
            'hemisphere': 'left',
            'angle': 'lateral_pivot',
            'surface': 'inflated'
        }
    }, 
        
    {
        'extent': [0.5, 0.05, 0.48, 0.48],
        'view': {
            'hemisphere': 'right',
            'angle': 'bottom_pivot',
            'surface': 'inflated'
        }
    }, 
    {
        'extent': [0.5, 0.4, 0.48, 0.48],
        'view': {
            'hemisphere': 'right',
            'angle': 'lateral_pivot',
            'surface': 'inflated'
        }
    },        
    ]
}


def get_single_view_params(view, surface='inflated', figsize=9):
    params = {
    'figsize': [figsize, figsize],
    'panels': [
    {
        'extent': [0.000, 0.1, 0.92, 1],
        'view': {
            'angle': view,
            'surface': surface,
        }
    },         
    ]
    }
    return params

def masked_vertex_nans(subject, vx):
    """
    Utility function to mask NaN values in vertex data
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'fsaverage')
    vx : cortex.Vertex
        Vertex data object to mask
        
    Returns
    -------
    cortex.VertexRGB
        RGB vertex data with NaN values masked by curvature
    """
    # Get curvature
    curv = cortex.db.get_surfinfo(subject)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map. 
    curv.vmin = 0
    curv.vmax = 1
    curv.data[curv.data > 0] = 0.05
    curv.data[curv.data < 0] = 0.4
    curv.cmap = 'binary'# Get curvature

    # Map to RGB
    vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])

    # Pick an arbitrary region to mask out
    # (in your case you could use np.isnan on your data in similar fashion)
    alpha = np.logical_not(np.isnan(vx.data))
    alpha = alpha.astype(np.float)

    # Alpha mask
    display_data = vx_rgb * alpha + curv_rgb * (1-alpha)

    # Create vertex RGB object out of R, G, B channels
    vx_fin = cortex.VertexRGB(*display_data, subject)
    
    return vx_fin


def get_vtc_masks(subject='fsaverage', concatenate=False, exclusionary=False):
    """
    Get masks for ventral temporal cortex (VTC) regions.
    
    Parameters
    ----------
    subject : str, optional
        Subject ID, default 'fsaverage'
    concatenate : bool, optional
        Whether to concatenate left and right hemisphere masks
    exclusionary : bool, optional
        If True, return indices outside VTC regions
        
    Returns
    -------
    dict or array
        Dictionary of hemisphere masks if concatenate=False,
        otherwise concatenated array of mask indices
    """
    hemi_masks = {}
    for hemi, hemisphere in {'L':'lh', 'R':'rh'}.items():
        labels, _, names = nib.freesurfer.read_annot(f'{BIDS_DIR}/derivatives/freesurfer/{subject}/label/{hemisphere}.aparc.annot')
        inds = [names.index(b'fusiform'),names.index(b'inferiortemporal')]
        if exclusionary:
            hemi_masks[hemi] = np.array([ii for ii, label in enumerate(labels) if label not in inds])
        else:
            hemi_masks[hemi] = np.array([ii for ii, label in enumerate(labels) if label in inds])
        if hemi == 'L':
            lh_len = len(labels)

    if concatenate:
        hemi_masks = np.concatenate((hemi_masks['L'], lh_len+hemi_masks['R']), axis=0)

    return hemi_masks

def default_pane_fn(maptype, nsubs, view, map_, plot_roi_labels, 
        fig_dir=os.path.join(FIGS_DIR, 'paper/panels'),
    ):
    """
    Generate default filename for brain visualization panels.
    
    Parameters
    ----------
    maptype : str
        Type of map ('hist' or other)
    nsubs : int
        Number of subjects
    view : str
        View type (e.g., 'LFV', 'LF')
    map_ : str
        Map name
    plot_roi_labels : bool
        Whether ROI labels are plotted
    fig_dir : str, optional
        Output directory
        
    Returns
    -------
    str
        Generated filename
    """
    nsub_tag = f'_nsubs-{nsubs}'
    ltag = '_roilabels' if plot_roi_labels else ''

    fn = os.path.join(fig_dir, f'fsaverage_gr-{maptype}{nsub_tag}_sub-logp_view-{view}_floc_{map_}_sm-4mm_M{ltag}.png')

    return fn

def plot_panes(subject, volume, map_, vmin, vmax, 
        show=True, 
        port=None, 
        plot_roi_labels=False, 
        view='LFV', 
        cmap='cmr.rainforest', 
        cmap_label=None,
        maptype='hist',
        nsubs=len(SUBNUMS),
        fn='default',
        decoding=False,
        acc_thresh=0.65,
        ):
    """
    Plot brain visualization panels with customizable views and colormaps.
    
    Parameters
    ----------
    subject : str
        Subject ID
    volume : cortex.Volume
        Volume data to plot
    map_ : str
        Map name
    vmin : float
        Minimum value for colormap
    vmax : float
        Maximum value for colormap
    show : bool, optional
        Whether to display plot
    port : int, optional
        Port for viewer
    plot_roi_labels : bool, optional
        Whether to show ROI labels
    view : str, optional
        View type ('LFV', 'LF', 'L', 'LV', 'ILMV', 'V', 'B')
    cmap : str, optional
        Colormap name
    cmap_label : str, optional
        Colormap label
    maptype : str, optional
        Type of map
    nsubs : int, optional
        Number of subjects
    fn : str, optional
        Output filename
    decoding : bool, optional
        Whether plot shows decoding results
    acc_thresh : float, optional
        Accuracy threshold for decoding results
        
    Returns
    -------
    tuple
        (figure, axes)
    """
    if fn == 'default':
        fn = default_pane_fn(maptype, nsubs, view, map_, plot_roi_labels)

    if view == 'LFV':
        params = params_lateral_flatmap_ventral
    elif view == 'LF':
        params = params_lateral_flatmap
    elif view == 'L':
        params = params_lateral
    elif view == 'LV':
        params = params_lateral_ventral
    elif view == 'ILMV':
        params = cortex.export.params_inflated_lateral_medial_ventral
    elif view == 'V':
        params = get_single_view_params('ventral')
    elif view == 'B':
        params = get_single_view_params('bottom_r')
    else:
        raise ValueError()

    if plot_roi_labels:
        labels_visible=['rois']
    else:
        labels_visible=['']
        
    fig = cortex.export.plot_panels(volume, **params, sleep=10, close=port is None,
        viewer_params=dict(labels_visible=labels_visible, overlays_visible=['rois'], port=port),
    )

    if cmap_label is None:
        if decoding:
            cmap_label = f'Subjects (acc>{acc_thresh})'
        else:
            cmap_label = 'Subjects (p<0.0001)' if maptype == 'hist' else 'Mean of subject log(p)'

    # fig.subplots_adjust(bottom=0.02)
    width, height = 0.4, 0.02
    ax = plt.axes((0.5 - width/2, 0.1, width, height))
    if cmap is not None:
        try:
            cmap = plt.get_cmap(cmap)
        except:
            cmap = plt.get_cmap(f'cmr.{cmap}')
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=ax, orientation='horizontal', 
                    label=cmap_label,
                    )

    if fn is not None:
        fig.savefig(fn, dpi=300, bbox_inches='tight', transparent=True)
    if show:
        plt.show()
    else:
        plt.close()

def crop_im(figname, qual=1.5):
    """
    Crop a flatmap image to show just the VTC part.
    
    Parameters
    ----------
    figname : str
        Input figure filename
    qual : float, optional
        Quality/scaling factor for cropping dimensions
        
    Returns
    -------
    tuple
        (cropped image array, output filename)
    """
    cropped_figname = f'{os.path.dirname(figname)}/cropped_{os.path.basename(figname)}'
    image = imageio.imread(figname)
    title_qual = qual

    startrow, endrow, startcol, endcol = 204, 627, 384, 926

    startrow, endrow, startcol, endcol = int(round(qual*startrow)), int(round(qual*endrow)), int(round(qual*startcol)), int(round(qual*endcol))
    title_end = int(round(title_qual*30))
    cropped_im = np.concatenate((image[0:title_end,startcol:endcol,:], image[startrow:endrow,startcol:endcol,:]), axis=0)

    return cropped_im, cropped_figname

def swap_axes(func_map):
    """
    Swap axes for correct viewing of nifti volume in pycortex.
    
    Parameters
    ----------
    func_map : array-like
        Functional map data
        
    Returns
    -------
    array-like
        Map with swapped axes
    """
    new_map = func_map.copy()
    new_map.setflags(write=True)
    new_map = new_map.swapaxes(0,2)
    return new_map

def sub_vol_to_fsaverage(subnum, in_file=None, in_data=None, overwrite=False, remove_out_file=False):
    """
    Convert subject volume to fsaverage space.
    
    Parameters
    ----------
    subnum : int
        Subject number
    in_file : str, optional
        Input filename
    in_data : array-like, optional
        Input data (alternative to in_file)
    overwrite : bool, optional
        Whether to overwrite existing output
    remove_out_file : bool, optional
        Whether to remove output files after loading
        
    Returns
    -------
    array-like
        Data transformed to fsaverage space
        
    Notes
    -----
    Either in_file or in_data must be provided, but not both.
    """
    assert in_file is not None or in_data is not None and not (in_file is not None and in_data is not None)
    if in_file is None:
        nib.save(nib.nifti1.Nifti1Image(in_data, np.eye(4)), 'tmp.nii')
        in_file = 'tmp.nii'
    folder = os.path.dirname(in_file)
    pattern = os.path.basename(in_file)
    out_name = in_file.replace('.nii.gz', '').replace('.nii', '') + '_fsaverage_M.gii'
    if os.path.exists(out_name) and not overwrite:
        return nib.load(out_name).darrays[0].data
    else:
        owd = os.getcwd()
        print(owd)
        if not len(folder):
            folder = owd
        base_dir = os.path.dirname(os.path.realpath(__file__)).replace('/func','/scripts').replace('/notebooks', '/scripts')
        overwrite_tag = '-o' if overwrite else ''
        print(base_dir)

        os.chdir(base_dir)
        for zipped in [True, False]:
            if not zipped:
                pattern = pattern.replace('.gz', '')
            command = f'bash sub_maps_to_fsaverage.sh -s {subnum:02d} -f {folder} -t {pattern} {overwrite_tag}'
            print(command)
            os.system(command)
        os.chdir(owd)

        out_data = nib.load(out_name).darrays[0].data

        if in_file == 'tmp.nii':
            os.remove('tmp.nii')
            for hemi in ['L','R', 'M']:
                os.remove(f'tmp_fsaverage_{hemi}.gii')
        elif remove_out_file:
            for hemi in ['L','R', 'M']:
                os.remove(out_name.replace('fsaverage_M', f'fsaverage_{hemi}'))        
    return out_data