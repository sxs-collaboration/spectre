#!/bin/bash -
#SBATCH -o spectre.out
#SBATCH -e spectre.out
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=24
#SBATCH -A sxs
#SBATCH --no-requeue
#SBATCH -J MyJobName
#SBATCH --nodes 4
#SBATCH -t 0:15:00

# Distributed under the MIT License.
# See LICENSE.txt for details.

# To run a job on Wheeler:
# - Set the -J, --nodes, and -t options above, which correspond to job name,
#   number of nodes, and wall time limit in HH:MM:SS, respectively.
# - Set the build directory, run directory, executable name,
#   and input file below. The input file path is relative to ${RUN_DIR}.
#
# NOTE: The executable will not be copied from the build directory, so if you
#       update your build directory this file will use the updated executable.
#
# Optionally, if you need more control over how SpECTRE is launched on
# Wheeler you can edit the launch command at the end of this file directly.
#
# To submit the script to the queue run:
#   sbatch Wheeler.sh

export SPECTRE_BUILD_DIR=/path/to/build/dir
export RUN_DIR=/panfs/ds08/sxs/run/dir
export SPECTRE_EXECUTABLE=EvolveScalarWave3D
export SPECTRE_INPUT_FILE=./Input3DPeriodic.yaml

export PATH=${SPECTRE_BUILD_DIR}/bin:$PATH

CHARM_PPN=$(expr ${SLURM_CPUS_PER_TASK} - 1)

echo "###################################"
echo "######       JOB INFO        ######"
echo "###################################"
echo
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run Directory: ${RUN_DIR}"
echo "Submit Directory: ${SLURM_SUBMIT_DIR}"
echo "Queue: ${SLURM_JOB_PARTITION}"
echo "Nodelist: ${SLURM_JOB_NODELIST}"
echo "Tasks: ${SLURM_NTASKS}"
echo "CPUs per Task: ${SLURM_CPUS_PER_TASK}"
echo "Charm ppn: ${CHARM_PPN}"
echo "PATH: ${PATH}"
echo

/usr/bin/modulecmd bash list

############################################################################
# Set desired permissions for files created with this script
umask 0022

echo
echo "###################################"
echo "######   Executable Output   ######"
echo "###################################"
echo

cd ${RUN_DIR}

mpirun -n ${SLURM_NTASKS} \
       ${SPECTRE_EXECUTABLE} ++ppn ${CHARM_PPN} +setcpuaffinity \
       --input-file ${SPECTRE_INPUT_FILE}
