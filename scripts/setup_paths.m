function success = setup_paths()
success = 0;
addpath(genpath('../external/bids_gen'))
addpath(genpath('../external/matlab_modules'))
addpath(genpath('../external/fLoc/functions'))
addpath(genpath('../func/matlab'))
spm_rmpath; % some conflicting functions from cosmomvpa need to be removed
addpath('../external/matlab_modules/spm12'); % cleanly add spm12
success = 1;

end
