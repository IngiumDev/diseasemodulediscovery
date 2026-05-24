process DRUGPRIORITIZATIONOFFLINENETMEDPY {
    tag "$meta.id.$algorithm"
    label 'process_medium'
    container 'ghcr.io/greenarchitect1/modulediscovery_python_dependencies:main'

    input:
        tuple val(meta), path(module), path(netmedpy_ppi), path(netmedpy_drug_targets), path(netmedpy_distances), path(drug_background), val(algorithm)
        val result_size

    output:
        tuple val(meta), val(algorithm), path("${meta.id}.${algorithm}.csv"), emit: drug_rankings

    when:
        task.ext.when == null || task.ext.when

    script:
    """
    drug_prioritization_netmedpy.py \
        --ppi "${netmedpy_ppi}" \
        --drug-targets "${netmedpy_drug_targets}" \
        --distances "${netmedpy_distances}" \
        --drug-background "${drug_background}" \
        --module "${module}" \
        --algorithm "${algorithm}" \
        --prefix "${meta.id}" \
        --result-size "${result_size}" \
        --n-processors $task.cpus \
        -l DEBUG
    """
}
