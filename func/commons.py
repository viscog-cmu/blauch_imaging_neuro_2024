import datetime
import numpy as np
import os

COLORS=['#347fff', '#00bac7', '#febe08']
BIDS_DIR = os.getenv('BIDS') + '/lateralization'

def get_name(opts, exclude_keys=[], include_keys=[], bool_keys=[], no_datetime=False, ext='.pkl'):
    """
    Generate a filename from a dictionary of options, with configurable key filtering and formatting.

    Parameters
    ----------
    opts : dict
        Dictionary of options to include in filename
    exclude_keys : list, optional
        Keys to exclude from the filename. Mutually exclusive with include_keys.
    include_keys : list, optional 
        Keys to include in the filename. Mutually exclusive with exclude_keys.
    bool_keys : list, optional
        Keys that should only be included if their value is True
    no_datetime : bool, optional
        If False (default), prepend datetime string to filename
    ext : str, optional
        File extension to append, default '.pkl'. Set to None for no extension.

    Returns
    -------
    str
        Generated filename with options encoded as key-value pairs separated by 
        underscores and hyphens. Format is:
        [datetime_]key1-val1_key2-val2_...[.ext]
    """
    name = None
    if not no_datetime:
        name = str(datetime.now()).replace(' ', '-').split('.')[0]

    if len(exclude_keys) > 0 and len(include_keys) > 0:
        raise ValueError('you tried to use both exclude_keys and include_keys, but they are mutually exclusive')

    for key in sorted(opts):
        if len(exclude_keys) > 0:
            if key in bool_keys:
                include = key not in exclude_keys and opts[key]
            else:
                include = key not in exclude_keys
        elif len(include_keys) > 0:
            if key in bool_keys:
                include = key in include_keys and opts[key]
            else:
                include = key in include_keys
        if include:
            if name is None:
                name = '-'.join((key, str(opts[key])))
            else:
                name = '_'.join((name, '-'.join((key, str(opts[key])))))
    if ext is not None:
        name += ext
    return name