{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Supercomputer at the Max Planck Computing Data Facility.
# More information:
# https://docs.mpcdf.mpg.de/doc/computing/clusters/systems/Gravitational_Physics_ACR.html

{% block head %}
{{ super() -}}
#SBATCH -D ./
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240000
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
#SBATCH -p {{ queue | default("p.debug") }}
{% endblock %}

{% block charm_ppn %}
# One thread for communication
CHARM_PPN=$(expr ${SLURM_CPUS_PER_TASK} - 2)
{% endblock %}

{% block list_modules %}
# Load compiler and MPI modules with explicit version specifications,
# consistently with the versions used to build the executable.
module purge
module load gcc/11
module load impi/2021.7
module load boost/1.79
module load gsl/1.16
module load cmake/3.26
module load hdf5-serial/1.12.2
module load anaconda/3/2021.11

# Load Spack environment
source /u/guilara/repos/spack/share/spack/setup-env.sh
spack env activate env3_spectre_impi

# Define Charm paths
export CHARM_ROOT=/u/guilara/charm_impi_2/mpi-linux-x86_64-smp
export PATH=$PATH:/u/guilara/charm_impi_2/mpi-linux-x86_64-smp/bin

# Load python environment
source $SPECTRE_HOME/env/bin/activate
{% endblock %}

{% block run_command %}
srun -n ${SLURM_NTASKS} ${SPECTRE_EXECUTABLE} \
    --input-file ${SPECTRE_INPUT_FILE} \
    ++ppn ${CHARM_PPN} +pemap 0-34,36-70 +commap 35,71 \
    ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}
