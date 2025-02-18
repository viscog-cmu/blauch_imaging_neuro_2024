import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
import sys
sys.path.append('.')
from func.hcp_plotting_funs import plot_pairwise_corrs, spearman_p_to_r, get_roi_from_atlas, get_sums
from func.hcp import ROIS_DICT, clean_contrast_name
from func.masking import RAJIMEHR_ROIS  

handedness = 'T'
laterality_style = 'laterality'

targ_rois = RAJIMEHR_ROIS

df = {'ROI_1':[], 'ROI_2':[], 'map_1':[], 'map_2':[], 'map1-map2':[], 'map1-map2_v2':[], 'r':[], 'p':[], 's':[], 'r,p':[], 'p_fdr':[], 'rejected':[], 'ceil':[], 'ceil_unc':[], 'hand_thresh':[]}

targ_contrasts = [('LANGUAGE', 'STORY'), ('SOCIAL', 'TOM-RANDOM')]
seed_contrasts = [('EMOTION','FACES-SHAPES'), ('WM','CUSTOM:FACE-PLACE'), ('WM','CUSTOM:TOOL-PLACE'), ('WM','CUSTOM:BODY-PLACE')]

for hand_thresh in tqdm(np.arange(-100,100,10)):
    seed_sums, seed_inds = get_sums(atlas='HCPMMP1', hand_thresh=hand_thresh, handednesses=[handedness])
    targ_sums, inds = get_sums(atlas='rajimehr', experiments=['SOCIAL', 'LANGUAGE'], hand_thresh=hand_thresh, handednesses=[handedness])
    for targ_task, targ_contrast in targ_contrasts:
        for seed_roi in ['FFC', 'VTC']:
            rois = ROIS_DICT[seed_roi] if seed_roi in ROIS_DICT else [seed_roi]
            for seed_task, seed_contrast in seed_contrasts:
                family_rs = []
                family_ps = []
                family_dofs = []
                for targ_roi in targ_rois.keys():
                    targ_laterality, _ = get_roi_from_atlas(targ_sums, targ_contrast, targ_task, handedness, [targ_roi], laterality_style, atlas='rajimehr')
                    seed_laterality, _ = get_roi_from_atlas(seed_sums, seed_contrast, seed_task, handedness, rois, laterality_style)
                    r, p = spearmanr(targ_laterality, seed_laterality, nan_policy='omit')
                    dof = np.sum(~np.isnan(targ_laterality) & ~np.isnan(seed_laterality))
                    seed_contrast_ = clean_contrast_name(seed_contrast)
                    targ_contrast_ = clean_contrast_name(targ_contrast)
                    df['ROI_1'].append(seed_roi)
                    df['ROI_2'].append(targ_roi)
                    df['map_1'].append(seed_contrast_)
                    df['map_2'].append(targ_contrast_)
                    df['map1-map2'].append(f'{seed_roi}: {seed_contrast_} \n ROI: {targ_contrast_}'.replace('Emotional', 'Em.'))
                    df['map1-map2_v2'].append(f'{seed_roi}: {seed_contrast_} vs. ROI: {targ_contrast_}')
                    df['r'].append(r)
                    df['p'].append(p)
                    df['s'].append(-np.sign(r)*np.log10(p))
                    df['hand_thresh'].append(hand_thresh)
                    family_rs.append(r)
                    family_ps.append(p)
                    family_dofs.append(dof)
                rejected, p_fdr = fdrcorrection(family_ps, alpha=0.05)
                df['rejected'].extend(rejected)
                df['p_fdr'].extend(p_fdr)
                for r_i, p_fdr_i, p_i, dof in zip(family_rs, p_fdr, family_ps, family_dofs):
                    p_str = r"$p_{FDR}=$" + f"{p_fdr_i:.04f}" if p_fdr_i>0.0001 else r"$p_{FDR}<0.0001$"
                    df['r,p'].append(f'r={r_i:.03f}\n{p_str}' if p_i < 0.05 else '')
                    ceil = spearman_p_to_r(0.05/len(family_rs), dof) # FWE
                    df['ceil'].append(ceil)
                    df['ceil_unc'].append(spearman_p_to_r(0.05, dof))
                
df = pd.DataFrame(df)
df['-ceil'] = -df['ceil']
df['-ceil_unc'] = -df['ceil_unc']
# make sure bars exist when corrs are tiny
df.loc[np.abs(df.r) < 0.005, 'r'] = np.sign(df[np.abs(df.r) < 0.005].r)*0.005

df.to_csv(f'data/rajimehr_hcp_long_range_corrs.csv', index=False)