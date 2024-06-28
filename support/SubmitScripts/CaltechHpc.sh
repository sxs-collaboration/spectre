{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Supercomputer at Caltech.
# More information:
# https://www.hpc.caltech.edu/documentation

# This submit script currently requests cascadelake nodes (with 56 cores per
# node). We can adjust it to request other types of nodes later.

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 28
#SBATCH -p {{ queue | default("expansion") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
{% if reservation is defined %}
#SBATCH --reservation={{ reservation }}
{% endif %}
{% endblock %}

{% block run_command %}
mpirun -n ${SLURM_NTASKS} \
  ${SPECTRE_EXECUTABLE} --input-file ${SPECTRE_INPUT_FILE} \
  ++ppn ${CHARM_PPN} +setcpuaffinity +no_isomalloc_sync \
  ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}
