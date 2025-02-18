# Individual variation in the functional lateralization of human ventral temporal cortex: Local competition and long-range coupling
Nicholas M. Blauch, David C. Plaut, Raina Vin, Marlene Behrmann; Individual variation in the functional lateralization of human ventral temporal cortex: Local competition and long-range coupling. Imaging Neuroscience 2025; doi: https://doi.org/10.1162/imag_a_00488

### To get set up:
- configure an environment variable `BIDS`, where you will store BIDS-formatted fMRI experiments, and `SERVERNAME`, which will be used to configure some other directories. If using bash, add this to you .bashrc file located at `~/.bashrc` (or `~/.bash_profile` on a mac):  
```bash
export BIDS=<path to directory containing BIDS experiment data>
```
- edit `func/__init__.py` and `func.hcp.py` to configure relevant paths

### Code info:

- `func/`  contains code for functional analysis  
- `func/heuristics/` contains heuristic files for heudiconv (dicom->BIDS)  
- `func/matlab/` contains matlab functions used in functional analysis  
- `scripts/` scripts that actually run analyses or save plots, using functions in `func` and `external`  
- `data/` processed data used to generate figures  
- `notebooks/`  one notebook for each main results figure in the paper  
- `external/` each subdirectory is a git submodule  
  - `external/bids_gen/` general code for bids formatted experiments
  - `external/fLoc` the experimental code used for the localizer experiments
  - `external/matlab_modules` general matlab dependencies in a common location  

### To plot the results of the paper 
- download the small (<200MB) fully processed data zip file and unpack into `data`
- check out the `notebooks`, which uses the processed results located in `data`

### To reproduce the results from scratch:

#### In-house experiment:
1. download the BIDS data from [Kilthub](doi.org/10.1184/R1/28378451)
2. Configure `func/__init__.py` to point to the directory holding the data as `BIDS_DIR`
3. Run `scripts/pipe_full.sbatch` for each subject. For this, you can use the submission script `scripts/pipe_full_batch.sh`
4. Run post-processing: `scripts/inhouse_postproc.sh` which will run a few different scripts

#### HCP:
1. Create a directory in which to place HCP pre-processed functional maps, and point to it with `HCP_DIR` in `func/hcp.py`.
2. Create a ConnectomeDB account and download the behavioral data (https://db.humanconnectome.org/data/projects/HCP_1200). Place this file in your `HCP_DIR`, with the file name `hcp_unrestricted_data.csv`, which we will use to determine the subject names. 
3. Get HCP AWS credentials for downloading the data with s3
4. Run the download script `python scripts/get_hcp_data.py`
5. Process data necessary for notebooks: run `scripts/hcp_postproc.sh` which will run a few different scripts

Feel free to email me (nblauch@gmail.com) or create an issue if you have issues running this or find any bugs. 
