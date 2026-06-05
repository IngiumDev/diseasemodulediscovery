process DRUGPRIORITIZATIONOFFLINEGT {
    tag "$meta.id.$algorithm"
    label 'process_single'
    container 'ghcr.io/ingiumdev/modulediscovery_python_dependencies:main'
    memory = 3.GB
    time = 30.min

    input:
        tuple val(meta), path(module), path(drug_prioritization_graph), val(algorithm)
        val includeIndirectDrugs
        val result_size

    output:
        tuple val(meta), val(algorithm), path("${meta.id}.${algorithm}.csv"), emit: drug_rankings
        tuple val(meta), val(algorithm), path("${meta.id}.${algorithm}.drug_predictions.tsv"), emit: drug_predictions

    when:
        task.ext.when == null || task.ext.when

    script:
    def result_size_arg = result_size != '' ? "--result-size \"${result_size}\"" : ""
    """
    drug_prioritization_graphtool.py \
        --drug-prioritization-graph "${drug_prioritization_graph}" \
        --module "${module}" \
        --algorithm "${algorithm}" \
        --prefix "${meta.id}" \
        --ranking-output "${meta.id}.${algorithm}.csv" \
        --drug-predictions-output "${meta.id}.${algorithm}.drug_predictions.tsv" \
        ${result_size_arg} \
        ${includeIndirectDrugs ? '--includeIndirectDrugs' : ''} \
        -l DEBUG
    """
}
