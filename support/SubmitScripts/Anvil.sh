#!/bin/bash -
#SBATCH -o spectre.out
#SBATCH -e spectre.out
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=128
#SBATCH --no-requeue
#SBATCH -J SpectreJob
#SBATCH --nodes 2
#SBATCH -t 12:00:00
#SBATCH -p standard

# Distributed under the MIT License.
# See LICENSE.txt for details.

# To run a job on Anvil:
# - Set the -J, --nodes, and -t options above, which correspond to job name,
#   number of nodes, and wall time limit in HH:MM:SS, respectively.
# - Set the build directory, run directory, executable name,
#   and input file below. The input file path is relative to ${RUN_DIR}.
#
# NOTE: The executable will not be copied from the build directory, so if you
#       update your build directory this file will use the updated executable.
#
# Optionally, if you need more control over how SpECTRE is launched on
# Anvil you can edit the launch command at the end of this file directly.
#
# To submit the script to the queue run:
#   sbatch Anvil.sh

export SPECTRE_HOME=$PROJECT/$USER/spectre/spectre
export SPECTRE_BUILD_DIR=$SPECTRE_HOME/build
export SPECTRE_DEPS=$PROJECT/$USER/spectre/deps
export RUN_DIR=$PWD
export SPECTRE_EXECUTABLE=$RUN_DIR/EvolveGhBinaryBlackHole
export SPECTRE_INPUT_FILE=$RUN_DIR/BinaryBlackHole.yaml

module use $SPECTRE_DEPS/modules
source $SPECTRE_HOME/support/Environments/anvil_gcc.sh
spectre_load_modules

############################################################################
# Set desired permissions for files created with this script
umask 0022

export PATH=${SPECTRE_BUILD_DIR}/bin:$PATH
cd ${RUN_DIR}

CHARM_PPN=$(expr ${SLURM_CPUS_PER_TASK} - 1)
echo "Slurm tasks: ${SLURM_NTASKS}"
echo "Slurm cpus per task: ${SLURM_CPUS_PER_TASK}"
echo "Charm ppn: ${CHARM_PPN}"

module list

mpirun -n ${SLURM_NTASKS} \
  ${SPECTRE_EXECUTABLE} ++ppn ${CHARM_PPN} \
  +setcpuaffinity --input-file ${SPECTRE_INPUT_FILE} \
  >> ${RUN_DIR}/spectre.out 2>&1
