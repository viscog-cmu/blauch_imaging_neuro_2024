import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    allowed template fields - follow python string module:
    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    t1w = create_key(os.path.join('sub-{subject}','anat', 'sub-{subject}_T1w'))
    func = create_key(os.path.join('sub-{subject}','func', 'sub-{subject}_task-func_run-{item:02d}_bold'))
    ret = create_key(os.path.join('sub-{subject}', 'func', 'sub-{subject}_task-ret_run-{item:02d}_bold'))
    prf = create_key(os.path.join('sub-{subject}', 'func', 'sub-{subject}_task-prf_run-{item:02d}_bold'))
    floc = create_key(os.path.join('sub-{subject}', 'func', 'sub-{subject}_task-floc_run-{item:02d}_bold'))
    lang = create_key(os.path.join('sub-{subject}', 'func', 'sub-{subject}_task-lang_run-{item:02d}_bold'))
    # fmap_mag = create_key(os.path.join('sub-{subject}', 'fmap', 'sub-{subject}_magnitude'))
    # fmap_phase =  create_key(os.path.join('sub-{subject}', 'fmap', 'sub-{subject}_phasediff'))
    fmap = create_key(os.path.join('sub-{subject}', 'fmap', 'sub-{subject}_acq-phasediff_dir-PA_epi'))
    b0_PA = create_key(os.path.join('sub-{subject}', 'dwi', 'sub-{subject}_acq-B0_dir-PA'))
    b0_PA_TRACEW = create_key(os.path.join('sub-{subject}', 'dwi', 'sub-{subject}_acq-B0_dir-PA_TRACEW'))
    b500 = create_key(os.path.join('sub-{subject}', 'dwi', 'sub-{subject}_acq-dti20dxb500'))
    b1500 = create_key(os.path.join('sub-{subject}', 'dwi', 'sub-{subject}_acq-dti64dxb1500'))
    b3000 = create_key(os.path.join('sub-{subject}', 'dwi', 'sub-{subject}_acq-dti64dxb3000'))

    info = {t1w: [], func: [], ret: [], prf:[], floc: [], lang: [], fmap:[],
            b0_PA: [], b0_PA_TRACEW: [], b500: [], b1500: [], b3000: []}
    for idx, s in enumerate(seqinfo):
        if ('mprage' in s.dcm_dir_name.lower()):
            info[t1w] = [s.series_id]
        if ('floc' in s.dcm_dir_name.lower()) or ('lateralization' in s.dcm_dir_name.lower()):
            info[floc].append(s.series_id)
        if 'langloc' in s.dcm_dir_name.lower():
            info[lang].append(s.series_id)
        if 'opp-phase' in s.dcm_dir_name.lower() or 'phasediff' in s.dcm_dir_name.lower():
            info[fmap].append(s.series_id)
        if 'acq-b0' in s.dcm_dir_name.lower() and 'tracew' not in s.dcm_dir_name.lower():
            info[b0_PA].append(s.series_id)
        if 'tracew' in s.dcm_dir_name.lower():
            info[b0_PA_TRACEW].append(s.series_id)
        if 'b500' in s.dcm_dir_name.lower():
            info[b500].append(s.series_id)
        if 'b1500' in s.dcm_dir_name.lower():
            info[b1500].append(s.series_id)
        if 'b3000' in s.dcm_dir_name.lower():
            info[b3000].append(s.series_id)

    print(len(info[floc]))
    return info
