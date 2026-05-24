process PRECOMPUTENETMEDPYDISTANCES {
    tag "$meta.id"
    label 'process_medium'

    input:
        tuple val(meta), path(netmedpy_ppi)

    output:
        tuple val(meta), path("${meta.id}.netmedpy_shortest_paths.pkl.npz") , emit: netmedpy_distances

    when:
        task.ext.when == null || task.ext.when

    script:
    """
    precompute_netmedpy_distances.py \
        --ppi "${netmedpy_ppi}" \
        --prefix "${meta.id}" \
        --n-processors $task.cpus \
        -l DEBUG
    """
}
