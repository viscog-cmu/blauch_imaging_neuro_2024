
import os
import glob
import shutil
import pdb
import subprocess
from tqdm import tqdm
import pandas as pd
from func.hcp import HCP_DIR

save_dir = HCP_DIR
sub_sheet = pd.read_csv(f'{HCP_DIR}/hcp_unrestricted_data.csv')
exps_to_use = ['SOCIAL', 'LANGUAGE', 'WM', 'EMOTION']

os.makedirs(f'{save_dir}/func_files', exist_ok=True)
subjects = sub_sheet.Subject.to_numpy()

for space in ['']: # '' or  '_MSMALL'
    exp_subs = {'WM':0, 'LANGUAGE':0, 'EMOTION':0, 'MOTOR':0, 'RELATIONAL':0, 'SOCIAL':0}
    for sub in subjects:
        for exp in exps_to_use:
            if not os.path.exists(f'{save_dir}/subjects/{sub}/MNINonLinear/Results/tfMRI_{exp}/tfMRI_{exp}_hp200_s2_level2{space}.feat/{sub}_tfMRI_{exp}_level2_hp200_s2{space}.dscalar.nii'):
                if os.path.exists(f'{save_dir}/1200/{sub}_3T_tfMRI_{exp}_analysis_s2.zip') and space != '_MSMAll':
                    shutil.unpack_archive(f'{save_dir}/1200/{sub}_3T_tfMRI_{exp}_analysis_s2.zip', f'{save_dir}/subjects/')
                else:
                    print(f'trying to acquire {exp} func data from aws')
                    os.makedirs(f'{save_dir}/subjects/{sub}/MNINonLinear/Results/tfMRI_{exp}/tfMRI_{exp}_hp200_s2_level2{space}.feat', exist_ok=True)
                    command = f'aws s3 cp s3://hcp-openaccess/HCP_1200/{sub}/MNINonLinear/Results/tfMRI_{exp}/tfMRI_{exp}_hp200_s2_level2{space}.feat/{sub}_tfMRI_{exp}_level2_hp200_s2{space}.dscalar.nii {save_dir}/subjects/{sub}/MNINonLinear/Results/tfMRI_{exp}/tfMRI_{exp}_hp200_s2_level2{space}.feat/'
                    print(command)
                    os.system(command)
            if os.path.exists(f'{save_dir}/subjects/{sub}/MNINonLinear/Results/tfMRI_{exp}/tfMRI_{exp}_hp200_s2_level2{space}.feat/{sub}_tfMRI_{exp}_level2_hp200_s2{space}.dscalar.nii'):
                exp_subs[exp] += 1
                print(f'{save_dir}/subjects/{sub}/MNINonLinear/Results/tfMRI_{exp}/tfMRI_{exp}_hp200_s2_level2{space}.feat/{sub}_tfMRI_{exp}_level2_hp200_s2{space}.dscalar.nii')
                proc = subprocess.run(f'cp {save_dir}/subjects/{sub}/MNINonLinear/Results/tfMRI_{exp}/tfMRI_{exp}_hp200_s2_level2{space}.feat/{sub}_tfMRI_{exp}_level2_hp200_s2{space}.dscalar.nii {save_dir}/func_files/{sub}_tfMRI_{exp}_level2_hp200_s2{space}.dscalar.nii'.split(' '), capture_output=True)
                print(proc.stdout)
                print(proc.stderr)
            else:
                print(f'failed to acquire {exp} func data from aws')
                command = f'aws s3 ls s3://hcp-openaccess/HCP_1200/{sub}/MNINonLinear/Results/'
                print(command)
                proc = subprocess.run(command.split(' '), capture_output=True)
                exps = [item.replace('PRE ', '').replace(' ', '') for item in proc.stdout.decode().split('\n')]
                if f'tfMRI_{exp}/' in exps:
                    proc = subprocess.run(f'aws s3 ls s3://hcp-openaccess/HCP_1200/{sub}/MNINonLinear/Results/tfMRI_{exp}/'.split(' '), capture_output=True)
                    print(proc.stdout.decode())