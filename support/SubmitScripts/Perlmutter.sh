{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# DOE supercomputer at Lawrence Berkely National Laboratory
# More information:
# https://docs.nersc.gov/systems/perlmutter/

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --constraint cpu
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-task 32
#SBATCH -q {{ queue | default("regular") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
{% endblock %}

{% block run_command %}
srun -n ${SLURM_NTASKS} \
  ${SPECTRE_EXECUTABLE} --input-file ${SPECTRE_INPUT_FILE} \
  ++ppn ${CHARM_PPN} +setcpuaffinity \
  ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}
