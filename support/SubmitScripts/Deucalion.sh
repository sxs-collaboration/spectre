{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Deucalion - Supercomputer at RNCA – Rede Nacional de Computação Avançada (National Advanced Computing Network) installed on the Azurém campus of the University of Minho
# https://docs.macc.fccn.pt/deucalion/
# The x86 nodes are: 2× AMD EPYC 7742 64-core 2.25 GHz

{% block head %}
{{ super() -}}
## The sbatch command should set the following parameters.
## For example by adding them as extra parameters (-p) to Schedule.py
##SBATCH --account {{ account }}
##SBATCH --mail-user {{ mail_user }}
#SBATCH --mail-type ALL
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node {{ num_slurm_tasks | default(8) }}
#SBATCH --cpus-per-task=16
#SBATCH --time {{ time_limit | default("2-00:00:00") }}
#SBATCH --partition {{ queue | default("normal-x86") }}
{% endblock %}

{% block charm_ppn %}
# Two thread for communication
CHARM_PPN=$(expr ${SLURM_CPUS_PER_TASK} - 2)
{% endblock %}

{% block list_modules %}
# Load compiler and MPI modules with explicit version specifications,
# consistently with the versions used to build the executable.
source @CMAKE_SOURCE_DIR@/support/Environments/deucalion.sh
spectre_load_modules
{% endblock %}

{% block run_command %}
srun -n ${SLURM_NTASKS} \
    ${SPECTRE_PROFILING_PREFIX} \
    ${SPECTRE_EXECUTABLE} --input-file ${SPECTRE_INPUT_FILE} \
    ++ppn ${CHARM_PPN} +setcpuaffinity \
    ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}
