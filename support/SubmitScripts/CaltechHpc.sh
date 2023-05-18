#!/bin/bash -
#SBATCH -o spectre.stdout
#SBATCH -e spectre.stderr
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 28
#SBATCH -J KerrSchild
#SBATCH --nodes 2
#SBATCH -p any
#SBATCH -t 12:00:00
#SBATCH -D .

# Distributed under the MIT License.
# See LICENSE.txt for details.

# To run a job on Caltech HPC:
# - Set the -J, --nodes, and -t options above, which correspond to job name,
#   number of nodes, and wall time limit in HH:MM:SS, respectively.
# - Set an `#SBATCH -A` argument for the computing account id.
# - Or set --reservation=RES_NAME if there is a reservation available
# - Set the build directory, run directory, executable name,
#   and input file below.
#
# NOTE: The executable will not be copied from the build directory, so if you
#       update your build directory this file will use the updated executable.
#
# Optionally, if you need more control over how SpECTRE is launched on
# Caltech HPC you can edit the launch command at the end of this file directly.
#
# To submit the script to the queue run:
#   sbatch CaltechHpc.sh

# Replace these paths with the path to your build directory, to the source root
# directory, the spectre dependencies module directory, and to the directory
# where you want the output to appear, i.e. the run directory.
# E.g., if you cloned spectre in your home directory, set
# SPECTRE_BUILD_DIR to ${HOME}/spectre/build. If you want to run in a
# directory called "Run" in the current directory, set
# SPECTRE_RUN_DIR to ${PWD}/Run
export SPECTRE_BUILD_DIR=${HOME}/Codes/spectre/build_clang/
export SPECTRE_MODULE_DIR=${HOME}/Codes/spectre_deps/modules/
export SPECTRE_HOME=${HOME}/Codes/spectre/
export SPECTRE_RUN_DIR=${PWD}/Run

# Choose the executable and input file to run
# To use an input file in the current directory, set
# SPECTRE_INPUT_FILE to ${PWD}/InputFileName.yaml
export SPECTRE_EXECUTABLE=${SPECTRE_BUILD_DIR}/bin/EvolveGeneralizedHarmonic
export SPECTRE_INPUT_FILE=${PWD}/KerrSchild.yaml

# These commands load the relevant modules and cd into the run directory,
# creating it if it doesn't exist
source ${SPECTRE_HOME}/support/Environments/caltech_hpc_gcc.sh
module use ${SPECTRE_MODULE_DIR}
spectre_load_modules
module list

mkdir -p ${SPECTRE_RUN_DIR}
cd ${SPECTRE_RUN_DIR}

# Copy the input file into the run directory, to preserve it
cp ${SPECTRE_INPUT_FILE} ${SPECTRE_RUN_DIR}/

# Set desired permissions for files created with this script
umask 0022

# Set the path to include the build directory's bin directory
export PATH=${SPECTRE_BUILD_DIR}/bin:$PATH

CHARM_PPN=$(expr ${SLURM_CPUS_PER_TASK} - 1)

echo "###################################"
echo "######       JOB INFO        ######"
echo "###################################"
echo
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run Directory: ${SPECTRE_RUN_DIR}"
echo "Submit Directory: ${SLURM_SUBMIT_DIR}"
echo "Queue: ${SLURM_JOB_PARTITION}"
echo "Nodelist: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "CPUs per Task: ${SLURM_CPUS_PER_TASK}"
echo "Charm ppn: ${CHARM_PPN}"
echo "PATH: ${PATH}"
echo
echo "###################################"
echo "######   Executable Output   ######"
echo "###################################"
echo

mpirun -n ${SLURM_NTASKS} \
    ${SPECTRE_EXECUTABLE} ++ppn ${CHARM_PPN} +setcpuaffinity \
    --input-file ${SPECTRE_INPUT_FILE}
