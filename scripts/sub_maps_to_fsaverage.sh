#/bin/bash -f
usage="$0 [-s sub -f folder -t tag -o overwrite]
Convert volumetric searchlight/univariate maps to fsaverage for flat map visualization"

EXPNAME='lateralization'
export map_dir=$BIDS/${EXPNAME}/derivatives/matlab/spm/sub-${sub}/SPM-floc_vol-T1w_sm-4mm
export overwrite=1
export tag='*.nii*'
export exclude='mni'

OPTIND=1
while getopts "hs:f:t:e:o:" opt; do
  case $opt in
    s) sub=${OPTARG} ;;
    f) map_dir=${OPTARG} ;;
    o) overwrite=${OPTARG} ;;
    h) echo "$usage"; exit ;;
    t) tag=${OPTARG} ;;
    e) exclude=${OPTARG} ;;
  esac
done

echo 'sub:' $sub
echo 'dir:' ${map_dir}
echo 'tag:' ${tag}
echo 'exclude:' ${exclude}
echo 'overwrite:' ${overwrite}

export fs_dir=$BIDS/${EXPNAME}/derivatives/freesurfer/sub-$sub

# loop over list of unsmoothed .nii files in native space only
ls $map_dir/${tag} | grep -v ${exclude} | grep -v in_b0 | while read fullfname;
do
    fname=$(basename $fullfname)
    dirname=$(dirname $fullfname)
    export fname_L="${dirname}/${fname%%.*}"_fsaverage_L.gii
    export fname_R="${dirname}/${fname%%.*}"_fsaverage_R.gii
    export fname_M="${dirname}/${fname%%.*}"_fsaverage_M.gii
    fname=${fullfname}


    if [ ! -f "${fname_L}" ] || [ ! -f "${fname_R}" ] || [ $overwrite -eq 1 ]; then
      #first convert the volumes to surface and save in fsaverage space
      mri_vol2surf \
      --mov "${fname}" \
      --o "${fname_L}" \
      --regheader sub-$sub \
      --projfrac 0.5 \
      --reg $fs_dir/mri/register.dat \
      --hemi lh \
      --trgsubject fsaverage
      mri_vol2surf \
      --mov "${fname}" \
      --o "${fname_R}" \
      --regheader sub-$sub \
      --projfrac 0.5 \
      --reg $fs_dir/mri/register.dat \
      --hemi rh \
      --trgsubject fsaverage
    fi

    if [ ! -f "${fname_M}" ] || [ ${overwrite} -eq 1 ]; then
      python ../external/bids_gen/merge_fs_hemi_overlays.py --fname-L "${fname_L}" --fname-R "${fname_R}" --fname-M "${fname_M}"
    fi
done