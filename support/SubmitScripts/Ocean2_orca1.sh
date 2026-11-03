{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Orca-1 Partition on the Ocean2 Supercomputer at Cal State Fullerton hosted by
# Geoffrey Lovelace
# More information:
# Ask Geoffrey Lovelace for more information on Ocean2.

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node {{ num_slurm_tasks | default(1) }}
#SBATCH --cpus-per-task 20
#SBATCH -p {{ queue | default("orca-1") }}
#SBATCH -t {{ time_limit | default("01-00:00:00") }}
{% endblock %}

{% block mpi_configuration %}
export I_MPI_PIN=1
export I_MPI_PIN_RESPECT_CPUSET=1
export I_MPI_PIN_DOMAIN=${SLURM_CPUS_PER_TASK}:compact
export I_MPI_PIN_ORDER=compact
{% endblock %}

{# Intel MPI confines each rank to its CPU domain. Charm++'s automatic
   affinity does not respect that domain, so leave thread placement to Linux. #}
{% block charm_cpu_affinity %}
CHARM_CPU_AFFINITY=
{% endblock %}
