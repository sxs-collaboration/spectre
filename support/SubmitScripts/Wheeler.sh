#!/bin/bash -
#SBATCH -o spectre.out
#SBATCH -e spectre.out
#SBATCH --ntasks-per-node 24
#SBATCH -A sxs
#SBATCH --no-requeue
#SBATCH -J my_job_name
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

############################################################################
# Set desired permissions for files created with this script
umask 0022

export PATH=${SPECTRE_BUILD_DIR}/bin:$PATH
cd ${RUN_DIR}

# The 23 is there because Charm++ uses one thread per node for communication
srun -n ${SLURM_JOB_NUM_NODES} -c 24 \
     ${SPECTRE_EXECUTABLE} ++ppn 23 --input-file ${SPECTRE_INPUT_FILE}
