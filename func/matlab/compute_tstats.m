
function compute_tstats(sub,ses,task,runs,reg_tag,mode,space,sm_fwhm)
% Compute t-statistics
%
% Args:
%   sub: Subject ID (numeric or string)
%   ses: Session number (0 for no session)
%   task: Task name (e.g. 'floc')
%   runs: Array of run numbers to include
%   reg_tag: Tag for registration/preprocessing, set to '' for default
%   mode: Analysis mode ('surf' or 'vol')
%   space: Space for analysis ('mni' or other)
%   sm_fwhm: Smoothing kernel FWHM in mm
%
% Computes contrasts and t-statistics from SPM first-level analysis.
% Creates contrast maps, t-maps, log10(p) maps and Cohen's d maps.
% For surface analysis, converts results to GIFTI format.

if strcmpi(space,'mni')
    space = 'MNI152NLin2009cAsym';
end

sm_tag = [];
if sm_fwhm > 0
    sm_tag = sprintf('_sm-%dmm',sm_fwhm);
end

if ~ischar(sub)
    sub = sprintf('%02d',sub);
end

if ses==0
    sesfold = '';
else
    sesfold=sprintf('/ses-%02d',ses);
end

if strcmp(mode,'surf')
    hemispheres = {'-L','-R'};
else
    hemispheres = {''};
end

bids_dir = get_bids_dir('lateralization');

for hemi = 1:length(hemispheres)

    matlabbatch = [];
    model_dir = sprintf('%s/derivatives/matlab/spm/sub-%s%s/SPM-%s_%s-%s%s%s%s', ...
        bids_dir, sub, sesfold, task, mode, space, sm_tag, hemispheres{hemi}, reg_tag);

    matlabbatch{1}.spm.stats.con.spmmat = {[model_dir,'/SPM.mat']};
    
    short_names={};
    
    for subset = {'even', 'odd', 'odd2', 'all'}
        % for naming files
        if ~strcmp(subset{1}, 'all')
            subset_tag = ['-',subset{1}];
        else
            subset_tag = '';
        end
        if strcmp(task,'floc')
            runs_to_use = [];
            confounds_per_run = [];
            for run_i = 1:runs
                confounds_info = loadjson(sprintf('%s/derivatives/matlab/spm/multiple_regressors/sub-%s/sub-%s_task-%s_run-%02d%s_multiregressors_info.json', bids_dir, sub, sub, task, run_i, reg_tag));
                n_confounds = confounds_info.n_confounds;
                confounds_per_run(run_i) = n_confounds;
                if (mod(run_i, 2) && strcmp(subset{1},'odd')) || (mod(run_i, 2) && strcmp(subset{1}, 'odd2') && run_i < 5) || (~mod(run_i, 2) && strcmp(subset{1},'even')) || strcmp(subset{1},'all')
                    runs_to_use(run_i) = 1;
                else
                    runs_to_use(run_i) = 0;
                end
            end

            active_t = make_contrast([1 1 1 1 1 1], confounds_per_run, runs_to_use);
            active_f = [
            make_contrast([5 -1 -1 -1 -1 -1]/5, confounds_per_run, runs_to_use);...
            make_contrast([-1 5 -1 -1 -1 -1]/5, confounds_per_run, runs_to_use);...
            make_contrast([-1 -1 5 -1 -1 -1]/5, confounds_per_run, runs_to_use);...
            make_contrast([-1 -1 -1 5 -1 -1]/5, confounds_per_run, runs_to_use);...
            make_contrast([-1 -1 -1 -1 5 -1]/5, confounds_per_run, runs_to_use);...
            make_contrast([-1 -1 -1 -1 -1 5]/5, confounds_per_run, runs_to_use);
            ];

            matlabbatch{1}.spm.stats.con.consess{length(short_names)+1}.tcon.name = 'visually active';
            matlabbatch{1}.spm.stats.con.consess{length(short_names)+1}.tcon.convec = active_t;
            matlabbatch{1}.spm.stats.con.consess{length(short_names)+1}.tcon.sessrep = 'none';        
            short_names{length(short_names)+1} = ['vis-on-vs-off', subset_tag];

            %category vs. others
            n_cons = length(short_names);
            count = 0;
            for category = {'faces', 'characters', 'words', 'letters', 'invwords', 'objects'}
                category = category{1};
                if strcmp(category, 'faces')
                    seq = [6 0 -1 -1 -3 -1];
                elseif strcmp(category, 'characters')
                    seq = [-3 0 2 2 -3 2];
                elseif strcmp(category, 'objects')
                    seq = [-3 0 -1 -1 6 -1];
                elseif strcmp(category, 'words')
                    seq = [-1 0 2 0 -1 0];
                elseif strcmp(category, 'invwords')
                    seq = [-1 0 0 2 -1 0];
                elseif strcmp(category, 'letters')
                    seq = [-1 0 0 0 -1 2];
                end
                count = count + 1;
                short_names{n_cons+count} = lower(sprintf('uni-%s-vs-others-comb%s', category, subset_tag));
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.name =sprintf('%s > others (combined)', category);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.convec = make_contrast(seq, confounds_per_run, runs_to_use);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.sessrep = 'none';
            end

            % category vs. objects
            n_cons = length(short_names);
            count = 0;
            for category = {'faces', 'characters', 'words', 'letters', 'invwords'}
                category = category{1};
                if strcmp(category, 'faces')
                    seq = [1 0 0 0 -1 0];
                elseif strcmp(category, 'characters')
                    seq = [0 0 1 1 -3 1];
                elseif strcmp(category, 'words')
                    seq = [0 0 1 0 -1 0];
                elseif strcmp(category, 'invwords')
                    seq = [0 0 0 1 -1 0];
                elseif strcmp(category, 'letters')
                    seq = [0 0 0 0 -1 1];
                end
                count = count + 1;
                short_names{n_cons+count} = lower(sprintf('uni-%s-vs-objects-comb%s', category, subset_tag));
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.name =sprintf('%s > others (combined)', category);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.convec = make_contrast(seq, confounds_per_run, runs_to_use);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.sessrep = 'none';
            end

            % category vs. faces
            n_cons = length(short_names);
            count = 0;
            for category = {'objects', 'characters', 'words'}
                category = category{1};
                if strcmp(category, 'objects')
                    seq = [-1 0 0 0 1 0];
                elseif strcmp(category, 'characters')
                    seq = [-3 0 1 1 0 1];
                elseif strcmp(category, 'words')
                    seq = [-1 0 1 0 0 0];
                elseif strcmp(category, 'invwords')
                    seq = [-1 0 0 1 0 0];
                elseif strcmp(category, 'letters')
                    seq = [-1 0 0 0 0 1];
                end
                count = count + 1;
                short_names{n_cons+count} = lower(sprintf('uni-%s-vs-faces-comb%s', category, subset_tag));
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.name =sprintf('%s > others (combined)', category);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.convec = make_contrast(seq, confounds_per_run, runs_to_use);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.sessrep = 'none';
            end

            % % category vs. text
            n_cons = length(short_names);
            count = 0;
            for category = {'objects', 'faces'}
                category = category{1};
                if strcmp(category, 'objects')
                    seq = [0 0 -1 -1 3 -1];
                elseif strcmp(category, 'faces')
                    seq = [3 0 -1 -1 0 -1];
                end
                count = count + 1;
                short_names{n_cons+count} = lower(sprintf('uni-%s-vs-text-comb%s', category, subset_tag));
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.name =sprintf('%s > others (combined)', category);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.convec = make_contrast(seq, confounds_per_run, runs_to_use);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.sessrep = 'none';
            end

            % category vs. fixation
            n_cons = length(short_names);
            count = 0;
            for category = {'objects', 'characters', 'words', 'faces', 'letter_strings', 'inverted_words'}
                category = category{1};
                if strcmp(category, 'objects')
                    seq = [0 0 0 0 1 0];
                elseif strcmp(category, 'characters')
                    seq = [0 0 1/3 1/3 0 1/3];
                elseif strcmp(category, 'words')
                    seq = [0 0 1 0 0 0];
                elseif strcmp(category, 'inverted_words')
                    seq = [0 0 0 1 0 0];
                elseif strcmp(category, 'letter_strings')
                    seq = [0 0 0 0 0 1];
                elseif strcmp(category, 'faces')
                    seq = [1 0 0 0 0 0];
                end
                count = count + 1;
                short_names{n_cons+count} = lower(sprintf('uni-%s-vs-fixation%s', category, subset_tag));
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.name =sprintf('%s > fixation', category);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.convec = make_contrast(seq, confounds_per_run, runs_to_use);
                matlabbatch{1}.spm.stats.con.consess{n_cons+count}.tcon.sessrep = 'none';
            end
        else
            error('Chose an invalid task')
        end
    end

    matlabbatch{1}.spm.stats.con.delete = 1;
    n_contrasts = length(matlabbatch{1}.spm.stats.con.consess);
    spm_jobman('run',matlabbatch);

    if strcmp(mode,'surf')
        ext='.gii';
        for ii = 1:n_contrasts
            try
                spm_gii_and_dat_2_gii([model_dir,'/spmT_',sprintf('%04d',ii)],1)
            catch
                spm_gii_and_dat_2_gii([model_dir,'/spmF_',sprintf('%04d',ii)],1)
            end
        end
    else
        ext='.nii';
    end

    SPM = load([model_dir,'/SPM.mat']);
    SPM = SPM.SPM;
    t_df = SPM.xX.erdf;
    save(sprintf('%s/df.mat', model_dir), 't_df')
    for ii = 1:n_contrasts
         %rename file names to be comprehensible
        try
            copyfile(sprintf('%s/spmT_%04d%s',model_dir,ii,ext), sprintf('%s/%s%s',model_dir,short_names{ii},ext))
            copyfile(sprintf('%s/con_%04d%s',model_dir,ii,ext), sprintf('%s/%s_con%s',model_dir,short_names{ii},ext))
        catch
            copyfile(sprintf('%s/spmF_%04d%s',model_dir,ii,ext), sprintf('%s/%s%s',model_dir,short_names{ii},ext))
            copyfile(sprintf('%s/ess_%04d%s',model_dir,ii,ext), sprintf('%s/%s_con%s',model_dir,short_names{ii},ext))
        end
        % create log10p maps for thresholding
        tmap = load_untouch_nii(sprintf('%s/%s%s',model_dir,short_names{ii},ext));
        logpmap = tmap;
        signmap = sign(tmap.img);
        logpmap.img = -(signmap).*log10(tcdf(abs(tmap.img), t_df, 'upper'));

        % create cohen's d effect size maps, some simple math: 
        % se = sd/sqrt(df) = tmap / con (according to https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=spm;1c3ebff3.1506) ; d = con/sd ; d = con^2 / (tmap * sqrt(df))
        tmap = load_untouch_nii(sprintf('%s/%s%s',model_dir,short_names{ii},ext));
        con = load_untouch_nii(sprintf('%s/%s_con%s',model_dir,short_names{ii},ext));
        dmap = con;
        dmap.img = (con.img.^2)./(tmap.img * sqrt(t_df));
        save_untouch_nii(logpmap, sprintf('%s/%s_cohend%s',model_dir,short_names{ii},ext));

        % fix infinities 
        logpmap.img(abs(logpmap.img) > 60) = signmap(abs(logpmap.img)>60)*60;
        save_untouch_nii(logpmap, sprintf('%s/%s_log10p%s',model_dir,short_names{ii},ext));
    end
    
end
end

function seq = make_contrast(betas_contrast, confounds_per_run, runs_to_use)
    % make contrast vector
    % for each run, we have a contrast over [beta_0, beta_1, ..., beta_n, confound_1, confound_2, ..., confound_m]
    % the full contrast is these runs concatenated, plus mean regressors for each run
    if length(runs_to_use) ~= length(confounds_per_run)
        error('runs_to_use and confounds_per_run must be the same length')
    end
    seq = [];
    for ii = 1:length(confounds_per_run)
        if runs_to_use(ii)
            seq = [seq betas_contrast zeros(1,confounds_per_run(ii))];
        else
            seq = [seq zeros(1,length(betas_contrast)+confounds_per_run(ii))];
        end
    end
    seq = [seq zeros(1,length(confounds_per_run))];
end
