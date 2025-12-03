{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Supercomputer at the Max Planck Computing Data Facility.
# More information:
# https://docs.mpcdf.mpg.de/doc/computing/viper-user-guide.html

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node {{ num_slurm_tasks | default(1) }}
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=128
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
#SBATCH -p {{ queue | default("p.general") }}
{% endblock %}

{% block charm_ppn %}
# Two thread for communication
CHARM_PPN=$(expr ${SLURM_CPUS_PER_TASK} - 2)
{% endblock %}

{% block list_modules %}
source ${SPECTRE_HOME}/support/Environments/viper.sh
spectre_load_modules

{% endblock %}

{% block run_command %}
srun -n ${SLURM_NTASKS} ${SPECTRE_EXECUTABLE} \
    --input-file ${SPECTRE_INPUT_FILE} \
    ++ppn ${CHARM_PPN} +pemap 0-62,64-126 +commap 63,127 \
    ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}
