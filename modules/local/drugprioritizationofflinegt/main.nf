process DRUGPRIORITIZATIONOFFLINEGT {
    tag "$meta.id.$algorithm"
    label 'process_single'

    input:
        tuple val(meta), path(module), path(drug_prioritization_graph), val(algorithm)
        val includeIndirectDrugs
        val result_size

    output:
        tuple val(meta), val(algorithm), path("${meta.id}.${algorithm}.csv"), emit: drug_rankings

    when:
        task.ext.when == null || task.ext.when

    script:
    """
    drug_prioritization_graphtool.py \
        --drug-prioritization-graph "${drug_prioritization_graph}" \
        --module "${module}" \
        --algorithm "${algorithm}" \
        --prefix "${meta.id}" \
        --result-size "${result_size}" \
        ${includeIndirectDrugs ? '--includeIndirectDrugs' : ''} \
        -l DEBUG
    """
}
