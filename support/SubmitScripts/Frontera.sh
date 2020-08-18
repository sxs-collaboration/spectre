#!/bin/bash
#SBATCH -J MyJobName       # Job name
#SBATCH -o MyJobName.o%j   # Name of stdout output file
#SBATCH -e MyJobName.e%j   # Name of stderr error file
#SBATCH -p normal          # Queue (partition) name
#SBATCH -N 2               # Total # of nodes
#SBATCH -n 2               # Total # of tasks, must be number of nodes
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)
# #SBATCH --mail-type=all  # Send email at begin and end of job
# #SBATCH --mail-user=username@tacc.utexas.edu
# #SBATCH -A account       # Project/Allocation name (req'd if you have more than 1)

# Distributed under the MIT License.
# See LICENSE.txt for details.

# To run a job on Frontera:
# - Set the -J, -N, -n, and -t options above, which correspond to job name,
#   number of nodes, number of tasks, and wall time limit in HH:MM:SS.
# - Set the build directory, run directory, executable name,
#   and input file below. The input file path is relative to ${RUN_DIR}.
#
# NOTE: The executable will not be copied from the build directory, so if you
#       update your build directory this file will use the updated executable.
#
# Optionally, if you need more control over how SpECTRE is launched on
# Frontera you can edit the launch command at the end of this file directly.
#
# To submit the script to the queue run:
#   sbatch Frontera.sh

# UPDATE THESE!!!
export SPECTRE_EXECUTABLE=./bin/ParallelInfo
export SPECTRE_INPUT_FILE=./Test.yaml

# Print out diagnostic info
module list 2>&1
pwd
date

echo ""
echo ""
echo ""

export IBRUN_TASKS_PER_NODE=1

if [ -f ${SPECTRE_EXECUTABLE} ]; then
    if [ -f ${SPECTRE_INPUT_FILE} ]; then
        ibrun -n ${SLURM_JOB_NUM_NODES} \
            ${SPECTRE_EXECUTABLE} ++ppn 55 \
            --input-file ${SPECTRE_INPUT_FILE} 2>&1
    else
        echo "Could not find input file ${SPECTRE_INPUT_FILE}"
        exit 1
    fi
else
    echo "Could not find executable ${SPECTRE_EXECUTABLE}"
    exit 1
fi
