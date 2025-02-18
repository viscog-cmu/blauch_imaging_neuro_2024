declare -a subs=('01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '23' '24' '25' '26' '27' '28' '29' '31' '32' '33' '34' '35' '36' '37' '38' '39' '40' '41' '42' '43' '44' '45' '46' '47' '48' '49' '50' '51' '52' '53' '54' '55' '56' '57' '58')

fmriprep=0 # run fmriprep preprocessing
postproc=0 # create masks, etc. # 
func=0 # run glms, etc
analysis=0 # run higher analysis scripts
local=0 # if we don't want to submit as a job, set to 1

if [ $local -eq 1 ]; then
  for subnum in ${subs[@]}; do
   export SUBNUM=${subnum}
   export DO_FMRIPREP=${fmriprep}
   export DO_POSTPROC=${postproc}
   export DO_FUNC=${func}
   export DO_ANALYSIS=${analysis}
   bash scripts/pipe_full.sbatch
  done
else
  for subnum in ${subs[@]}; do
    sbatch --out=log/%j_sub-${subnum}.log \
        --export=SUBNUM=${subnum},DO_FMRIPREP=${fmriprep},DO_POSTPROC=${postproc},DO_FUNC=${func},DO_ANALYSIS=${analysis} \
        --job-name=sub-${subnum} \
        --cpus-per-task 4 --mem=8GB \
        scripts/pipe_full.sbatch
  done
fi