import numpy as np
import os
from .commons import BIDS_DIR

SUBNUMS_EXP1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29]
SUBNUMS_EXP2 = [31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,53,54,55,56,57,58]
SUBNUMS = np.concatenate((SUBNUMS_EXP1, SUBNUMS_EXP2))
# 4 and 50 excluded due to RH text lateralizaton. 20 and 31 excluded due to missing DWI data. 
SUBNUMS_PAPER = [sub for sub in SUBNUMS if sub not in [4, 20, 31, 50]]  
CONDITIONS = ['Faces','Inverted_Faces','Words','Inverted_Words','Objects','Letter_Strings']
CONDITIONS_NEW = ['Faces', 'Scrambled', 'Words', 'Inverted_Words', 'Objects', 'Letter_Strings']
USER = os.getenv('USER')

if os.getenv('SERVERNAME') == 'mind':
    FIGS_DIR = os.path.join(os.getenv('BIDS'), 'lateralization/figures', USER)
    HCP_DIR = '/lab_data/behrmannlab/scratch/hcp'
elif os.getenv('SERVERNAME') == 'xps':
    FIGS_DIR =  os.path.join(os.getenv('WHOME'), 'git/lateralization-overleaf/figures')
    HCP_DIR = '/mnt/d/hcp'
elif os.getenv('SERVERNAME') == 'mind_raina':
    FIGS_DIR = os.path.join(os.getenv('BIDS'), 'lateralization/figures/raina')
    HCP_DIR = '/lab_data/behrmannlab/scratch/hcp'
else:
    raise NotImplementedError('Need to set SERVERNAME env variable and/or configure default directories')

def X_is_running():
    """
    Check if X server is running.
    
    Returns
    -------
    bool
    """
    
    from subprocess import Popen, PIPE
    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0

def isnotebook():
    """
    Check if code is running in a notebook.
    
    Returns
    -------
    bool
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def check_matplotlib():
    """
    Check if matplotlib backend should be set to Agg.
    
    Returns
    -------
    bool
    """
    if not X_is_running() and not isnotebook():
        print('no X server detected. changing default matplotlib backend to Agg for compatibility.')
        import matplotlib
        matplotlib.use('Agg')