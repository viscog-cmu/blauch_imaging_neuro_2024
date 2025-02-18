#!/bin/bash -l

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=blauch@cmu.edu
#SBATCH -p cpu
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=24:00:00

cd ~/git/blauch_imag_neuro_2025/scripts

module load singularity
module load freesurfer-6.0.0
module load matlab-9.5

sub=$SUBNUM
export BIDS=/lab_data/behrmannlab/bids
export EXPNAME='lateralization'
export SUBJECTS_DIR=${BIDS}/${EXPNAME}/derivatives/freesurfer

export TEMPLATEFLOW_HOME=/home/nblauch/templateflow
unset PYTHONPATH

if [ ! -z "$SLURM_JOB_ID" ]; then
    export TMPDIR=$SCRATCH/$SLURM_JOB_ID
else
    export TMPDIR=/scratch/nblauch/pipe_full
fi
mkdir -p $TMPDIR

if [ $DO_FMRIPREP -eq 1 ]; then
    singularity run --bind /lab_data/behrmannlab:/lab_data/behrmannlab /user_data/nblauch/extras/fmriprep-1.4.1.simg \
    ${BIDS}/${EXPNAME} \
    ${BIDS}/${EXPNAME}/derivatives \
    participant \
    -w ${BIDS}/${EXPNAME}/derivatives/work \
    --fs-license-file ~/license.txt \
    --output-spaces anat MNI152NLin2009cAsym \
    --nthreads 12 --mem_mb 32000 \
    --ignore slicetiming \
    --participant-label ${sub} \
    --verbose
    rm -rf ${BIDS}/${EXPNAME}/derivatives/fmriprep/sub-${sub}/func/*space-MNI152NLin2009cAsym*
    rm -rf ${BIDS}/${EXPNAME}/derivatives/fmriprep/sub-${sub}/func/*space-fsaverage5*
    
fi

if [ $DO_POSTPROC -eq 1 ]; then

    #gunzip for SPM ...
    yes n | gunzip -r ${BIDS}/${EXPNAME}/derivatives/fmriprep/sub-${sub}
    
    #--------------------------------------get cortex (full/LH/RH) masks---------------------------------------
    if [ ! -f ${SUBJECTS_DIR}/sub-${sub}/mri/T1w.cortex.nii ]; then
        #binarize all voxels in ribbon: lh and rh
        mri_binarize --i ${SUBJECTS_DIR}/sub-${sub}/mri/ribbon.mgz \
            --o ${SUBJECTS_DIR}/sub-${sub}/mri/cortex.mgz --match 3 --match 42
        #convert to functional space
        mri_vol2vol --mov ${SUBJECTS_DIR}/sub-${sub}/mri/cortex.mgz \
            --regheader \
            --targ ${BIDS}/${EXPNAME}/derivatives/fmriprep/sub-${sub}/func/sub-${sub}_task-floc_run-01_space-T1w_desc-preproc_bold.nii \
            --o ${SUBJECTS_DIR}/sub-${sub}/mri/T1w.cortex.nii

        mri_binarize --i ${SUBJECTS_DIR}/sub-${sub}/mri/ribbon.mgz \
            --o ${SUBJECTS_DIR}/sub-${sub}/mri/lh.cortex.mgz --match 3
        #convert to functional space
        mri_vol2vol --mov ${SUBJECTS_DIR}/sub-${sub}/mri/lh.cortex.mgz \
            --regheader \
            --targ ${BIDS}/${EXPNAME}/derivatives/fmriprep/sub-${sub}/func/sub-${sub}_task-floc_run-01_space-T1w_desc-preproc_bold.nii \
            --o ${SUBJECTS_DIR}/sub-${sub}/mri/T1w.lh.cortex.nii

        mri_binarize --i ${SUBJECTS_DIR}/sub-${sub}/mri/ribbon.mgz \
            --o ${SUBJECTS_DIR}/sub-${sub}/mri/rh.cortex.mgz --match 42
        #convert to functional space
        mri_vol2vol --mov ${SUBJECTS_DIR}/sub-${sub}/mri/rh.cortex.mgz \
            --regheader \
            --targ ${BIDS}/${EXPNAME}/derivatives/fmriprep/sub-${sub}/func/sub-${sub}_task-floc_run-01_space-T1w_desc-preproc_bold.nii \
            --o ${SUBJECTS_DIR}/sub-${sub}/mri/T1w.rh.cortex.nii
    fi

    #create register.dat file: func -> freesurfer vol
    if [ ! -f ${SUBJECTS_DIR}/sub-${sub}/mri/register.dat ]; then
        tkregister2 --mov ${SUBJECTS_DIR}/sub-${sub}/mri/T1w.cortex.nii \
            --fstarg --regheader --reg ${SUBJECTS_DIR}/sub-${sub}/mri/register.dat --s sub-${sub} --noedit
    fi
    
    # create aparc+aseg file in func space
    if [ ! -f ${SUBJECTS_DIR}/sub-${sub}/mri/T1w.aparc+aseg.nii ]; then
        mri_label2vol --seg ${SUBJECTS_DIR}/sub-${sub}/mri/aparc+aseg.mgz \
            --temp ${BIDS}/${EXPNAME}/derivatives/fmriprep/sub-${sub}/func/sub-${sub}_task-floc_run-01_space-T1w_desc-preproc_bold.nii \
            --o ${SUBJECTS_DIR}/sub-${sub}/mri/T1w.aparc+aseg.nii --regheader
    fi

    echo $PWD
    #-------------------------------------merge inflated hemispheres for easier viewing in freeview----------------------------
    matlab -nodisplay -nosplash -r "setup_paths; \
        merge_fs_hemi_surfs('${EXPNAME}', 'sub-${sub}', 'inflated'); \
        merge_fs_hemi_curvs('${EXPNAME}', 'sub-${sub}'); \
        merge_fs_hemi_annotations('${EXPNAME}', 'sub-${sub}', 'aparc'); \
        exit"

    #-------------------------------------setup pycortex subject folder----------------------------
    cd ../external/bids_gen
    conda deactivate
    conda activate pycortex
    python setup_pycortex.py --subnum ${sub} --experiment-id ${EXPNAME} --experiment-short lat
    cd ../../scripts

fi


if [ $DO_FUNC -eq 1 ]; then
    echo $PWD
    #------------------------------------functional setup--------------------------------------------
    matlab -nodisplay -nosplash -r "\
        setup_paths; \
        bids_make_spm_multi_regression_files("${sub}",'${EXPNAME}',{'floc'},1); \
        make_spm_behav_conditions("$sub"); \
        exit"

    #-------------------------------------GLMs and other functional analyses----------------------------
    matlab -nodisplay -nosplash -r "\
        setup_paths; \
        do_vol_glm("${sub}", 'floc', 'T1w', 0); do_vol_glm("${sub}", 'floc', 'T1w', 4); \
        compute_tstats("${sub}", 0, 'floc', 5, '', 'vol', 'T1w', 4); \
        exit"
    matlab -nodisplay -nosplash -r "\
        setup_paths; \
        get_func_tstats("${sub}", 'T1w', 0, '${EXPNAME}', 'floc'); \
        save_func_ds(["${sub}"], 'T1w', 1, 1, 0, '${EXPNAME}', 'floc', 0); \
        exit"

    #-------------------------------------convert maps to fsaverage----------------------------
    source sub_maps_to_fsaverage.sh -s ${sub}
    source sub_maps_to_fsaverage.sh -s ${sub} -o 1 -t *log10p*.nii -f $BIDS/${EXPNAME}/derivatives/matlab/spm/sub-${sub}/SPM-floc_vol-T1w_sm-4mm

fi

#--------------------------------------higher analysis scripts--------------------------------
if [ $DO_ANALYSIS -eq 1 ]; then
    cd ..
    python scripts/vtc_ind_diffs.py --sub ${sub}
    cd scripts
fi

# Cleanup local work directory
rm -rf TMPDIR