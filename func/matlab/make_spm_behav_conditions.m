function make_spm_behav_conditions(subjects)
% Creates SPM behavioral condition files for fMRI analysis
%
% Args:
%   subjects: Array of subject IDs (numeric)
%
% Creates behavioral condition files in BIDS derivatives directory for SPM analysis.
% Currently supports fLoc task. Extracts timing information from raw behavioral
% files and creates condition files with onsets, durations and names for each run.
% This files are necessary for SPM first-level analysis.

    
bids_dir = [getenv('BIDS'),'/lateralization'];
tasks = {'floc'};
scanner_countdown = 0; %at BRIDGE, countdown automatically removed
session_countdown = 4;

%make sure floc class can be utilized
addpath('../experimental/fLoc/functions/classes')

for sub_i = 1:length(subjects)
    sub = subjects(sub_i);
    behav_dir = sprintf('%s/derivatives/matlab/spm/behavioral/sub-%02d',bids_dir,sub);
    if ~exist(behav_dir,'dir')
        mkdir(behav_dir)
    end
    for task_i = 1:length(tasks)
        task = tasks{task_i};
        clear('durations','names','onsets')
        switch task
            case 'floc'
                raw_behav_dir = sprintf('%s/raw_behavioral/floc/sub-%02d',bids_dir,sub);
                seq_file = dir([raw_behav_dir,'/*Sequence*']);
                seq = load([seq_file.folder,'/',seq_file.name]);
                try
                    seq=seq.sequence;
                catch
                    seq = seq.seq;
                end
                if sub >= 30
                    names = {'Faces', 'Scrambled', 'Words', 'Inverted_Words', 'Objects', 'Letter_Strings'};
                else
                    names = {'Faces', 'Inverted_Faces', 'Words', 'Inverted_Words', 'Objects', 'Letter_Strings'};
                end
                % dealing with some errors in launching the script at the scanner. sometimes the "at_scanner" parameter was accidentally changed instead of the "run_num" parameter.
                if sub == 40
                    loc_runs = [1,1,1,1,1];
                elseif sub == 42
                    loc_runs = [1,1,1,1,5];
                elseif sub == 43
                    tmp_dir = sprintf('%s/raw_behavioral/floc/sub-42',bids_dir);
                    seq_file = dir([tmp_dir,'/*Sequence*']);
                    tmp_seq = load([seq_file.folder,'/',seq_file.name]);
                    tmp_seq = tmp_seq.seq;
                    loc_runs = 1:seq.num_runs;
                elseif sub == 44
                    loc_runs = [5,5,5,4,5];
                else
                    loc_runs = 1:seq.num_runs;
                end
                loc_run_i = 0;
                for loc_run = loc_runs
                    if sub == 43
                        if loc_run == 1
                            normal_seq = seq;
                            seq = tmp_seq;
                        else
                            seq = normal_seq;
                        end
                    end
                    loc_run_i = loc_run_i + 1;
                    clear('durations','stim_names','onsets')
                    stim_names = cell(1,length(seq.stim_names));
                    for stim = 1:length(seq.stim_names)
                        if strncmp('face',seq.stim_names(stim,loc_run),4)
                            stim_names{stim} = 'Faces';
                        elseif strncmp('inverted_face',seq.stim_names(stim,loc_run),13)
                            stim_names{stim} = 'Inverted_Faces';
                        elseif strncmp('word',seq.stim_names(stim,loc_run),4)
                            stim_names{stim} = 'Words';
                        elseif strncmp('inverted_word',seq.stim_names(stim,loc_run),13)
                            stim_names{stim} = 'Inverted_Words';
                        elseif strncmp('object',seq.stim_names(stim,loc_run),6)
                            stim_names{stim} = 'Objects';
                        elseif strncmp('letter_string',seq.stim_names(stim,loc_run),13)
                            stim_names{stim} = 'Letter_Strings';
                        elseif strncmp('scrambled', seq.stim_names(stim,loc_run),9)
                            stim_names{stim} = 'Scrambled';
                        end
                    end
                    stim_durations(1:length(seq.stim_names)) = seq.stim_dur;
                    stim_onsets = (seq.stim_onsets(:,loc_run))';

                    onsets_trial = cell(1,length(names)); durations = cell(1,length(names));
                    for ii = 1:length(names)
                        onsets_trial{ii} = scanner_countdown + session_countdown + stim_onsets(strcmp(stim_names,names{ii}));
                        durations{ii} = stim_durations(strcmp(stim_names,names{ii}));
                    end

                    onsets = cell(1,length(names)); durations = cell(1,length(names));
                    for ii = 1:length(names)
                        onsets{ii} = onsets_trial{ii}(1:seq.stim_per_block:end);
                        durations{ii} = repmat(seq.stim_per_block*seq.stim_duty_cycle,1,length(onsets{ii}));
                    end

                    save(sprintf('%s/behav_conditions_%s-%02d.mat',behav_dir,task,loc_run_i),'durations','onsets','names')

                end
        end
    end
end
