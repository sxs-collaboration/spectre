{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Uranina -- HPC cluster of ACR division of MPI for Grav Physics, housed at the
# Max Planck Computing & Data Facility.
# https://docs.mpcdf.mpg.de/doc/computing/clusters/systems/Gravitational_Physics_ACR.html

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=72
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
#SBATCH -p {{ queue | default("p.urania") }}
{% endblock %}

{% block charm_ppn %}
# Two thread for communication
CHARM_PPN=$(expr ${SLURM_CPUS_PER_TASK} - 2)
{% endblock %}

{% block list_modules %}
# Load compiler and MPI modules with explicit version specifications,
# consistently with the versions used to build the executable.
source ${SPECTRE_HOME}/support/Environments/urania.sh
spectre_load_modules
spectre_setup_charm_paths

{% endblock %}

{% block run_command %}
srun -n ${SLURM_NTASKS} ${SPECTRE_EXECUTABLE} \
    --input-file ${SPECTRE_INPUT_FILE} \
    ++ppn ${CHARM_PPN} +pemap 0-34,36-70 +commap 35,71 \
    ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}
