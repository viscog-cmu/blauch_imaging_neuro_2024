function do_vol_glm( sub, task, space, sm_fwhm, reg_tag )
% Perform volume-based GLM analysis using SPM
%
% Args:
%   sub: Subject ID (numeric)
%   task: Task name (string), should be 'floc
%   space: Space for analysis ('mni' or other)
%   sm_fwhm: Smoothing kernel FWHM in mm (numeric)
%   reg_tag: Tag for registration/preprocessing (string, optional)
%
% Performs first-level GLM analysis on preprocessed fMRI data using SPM.
% Handles multiple sessions and runs. Creates SPM.mat files with model 
% specification for later contrast computation.
% Requires behavioral condition and multiple regressor files to be present.


clear('matlabbatch')
bids_dir = [getenv('BIDS'), '/lateralization'];
spm_dir = [bids_dir, '/derivatives/matlab/spm'];

% find all sessions or allow for non-session format
sessions = dir(sprintf('%s/derivatives/fmriprep/sub-%02d/*ses*', bids_dir, sub));
if isempty(sessions)
    sessions = {''};
else
    sessions = {sessions.name};
end

if strcmpi(space,'mni')
    space = 'MNI152NLin2009cAsym';
end

sm_tag = [];
if sm_fwhm > 0 
    sm_tag = sprintf('_sm-%dmm',sm_fwhm);
end

if nargin < 5
    reg_tag = '';
end

%ensure that jsonlab toolbox is on path if fails here
task_info = loadjson([bids_dir,'/task-',task,'_bold.json']);

% no special model IDs for this experiment
model_IDs = {''};

for ses = 1:length(sessions)
    session = sessions{ses};
    sesfold='';
    sestag='';
    if ~isempty(session)
        sesfold = ['/',session];
        sestag=['_',session];
    end
    behav_dir = sprintf('%s/behavioral/sub-%02d%s',spm_dir,sub,sesfold);
    preproc_dir =sprintf('%s/derivatives/fmriprep/sub-%02d%s/func',bids_dir,sub,sesfold);
    run_files = dir(sprintf('%s/derivatives/fmriprep/sub-%02d%s/func/*%s*%s*preproc_bold.nii', bids_dir, sub, sesfold,task,space));
    nruns = length(run_files);
    if nruns == 0
        continue
    end
    task_info_reprun = loadjson(sprintf('%s/sub-%02d%s/func/sub-%02d%s_task-%s_run-01_bold.json', bids_dir, sub, sesfold, sub, sestag, task));
    for model_ID = model_IDs
        %% specify
        clear matlabbatch
        clear SPM
        matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT = task_info.RepetitionTime;
        try
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = length(task_info.SliceTiming);
        catch
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = length(task_info_reprun.SliceTiming);
        end
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 1;
        model_dir = sprintf('%s/derivatives/matlab/spm/sub-%02d%s/SPM-%s%s_vol-%s%s%s',bids_dir,sub,sesfold,task,model_ID{1},space,sm_tag,reg_tag);
        if exist([model_dir,'/SPM.mat'],'file') && exist([model_dir,'/beta_0001.nii'],'file')
            load([model_dir,'/SPM.mat'])
             %if SPM is nonexistent or corrupted, deleted directory to rerun model
            if ~exist('SPM','var')
                rmdir(model_dir,'s')
            else
                [~,mask_name,~] = fileparts(SPM.xM.VM.fname);
                if ~isfield(SPM,'Vbeta') || ~contains(mask_name,[space,'_brainmask'])
                    rmdir(model_dir,'s')
                else
                    continue
                end
            end
        end
        matlabbatch{1}.spm.stats.fmri_spec.dir = {model_dir};
        for exp_run = 1:nruns
            % make sure smoothed files exist
            if sm_fwhm > 0
                scan = sprintf('%s/sub-%02d%s_task-%s_run-%02d_space-%s_desc-preproc_bold.nii',preproc_dir,sub,sestag,task,exp_run,space);
                [fdir,fname,ext] = fileparts(scan);
                sm_scan =  strcat(fdir,'/',fname,sm_tag,ext);
                smooth_masked_vols({scan}, {sm_scan}, repmat(sm_fwhm,[1,3]), [2 2 2]); %just an implicit NAN mask
            end
            scans = cellstr(spm_select('expand',sprintf('%s/sub-%02d%s_task-%s_run-%02d_space-%s_desc-preproc_bold%s.nii',preproc_dir,sub,sestag,task,exp_run,space,sm_tag)));
            matlabbatch{1}.spm.stats.fmri_spec.sess(exp_run).scans = scans;
            matlabbatch{1}.spm.stats.fmri_spec.sess(exp_run).multi = {sprintf('%s/behav_conditions_%s%s-%02d.mat',behav_dir,task,model_ID{1},exp_run)};
            matlabbatch{1}.spm.stats.fmri_spec.sess(exp_run).multi_reg = {sprintf('%s/multiple_regressors/sub-%02d%s/sub-%02d%s_task-%s_run-%02d%s_multiregressors.txt',spm_dir,sub,sesfold,sub,sestag,task,exp_run,reg_tag)};
            matlabbatch{1}.spm.stats.fmri_spec.sess(exp_run).hpf = 128;
        end

        matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh = -inf;
        matlabbatch{1}.spm.stats.fmri_spec.mask = {sprintf('%s/sub-%02d%s_task-%s_run-%02d_space-%s_desc-brain_mask.nii',preproc_dir,sub,sestag,task,exp_run,space)};
        matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

        spm_jobman('run',matlabbatch)

        %% estimate
        clear matlabbatch
        matlabbatch{1}.spm.stats.fmri_est.spmmat = {[model_dir,'/SPM.mat']};
        matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
        spm_jobman('run',matlabbatch)

        [~,model_name,~] = fileparts(model_dir);
        fprintf('estimated: sub-%d %s',sub,model_name)

    end
end
