process PREPAREDRUGPRIORITIZATIONINPUTS {
    tag "$meta.id"
    label 'process_single'
    container 'ghcr.io/greenarchitect1/modulediscovery_python_dependencies:main'

    input:
        tuple val(meta), path(ppi_gt)
        path drug_has_target_csv
        path drug_csv

    output:
        tuple val(meta), path("${meta.id}.drug_prioritization.gt") , emit: drug_prioritization_graph
        tuple val(meta), path("${meta.id}.drug_background.tsv")    , emit: drug_background
        tuple val(meta), path("${meta.id}.pdi.gt")                 , emit: pdi_graph
        tuple val(meta), path("${meta.id}.netmedpy_ppi.pkl", optional: true)          , emit: netmedpy_ppi
        tuple val(meta), path("${meta.id}.netmedpy_drug_targets.pkl", optional: true) , emit: netmedpy_drug_targets

    when:
        task.ext.when == null || task.ext.when

    script:
    """
    prepare_drug_prioritization_inputs.py \
        --ppi "${ppi_gt}" \
        --pdi "${drug_has_target_csv}" \
        --drugs "${drug_csv}" \
        --prefix "${meta.id}" \
        --id-space "${params.id_space}" \
        ${params.includeNonApprovedDrugs ? '--includeNonApprovedDrugs' : ''} \
        -l DEBUG
    """
}
