#!/bin/bash
#SBATCH -J MyJobName       # Job name
#SBATCH -o spectre.stdout  # Name of stdout output file
#SBATCH -e spectre.stderr   # Name of stderr error file
#SBATCH -p small           # Queue (partition) name - for 3+ nodes use 'normal'
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

# Replace these paths with the path to your build directory, to the source root
# directory, the spectre dependencies module directory, and to the directory
# where you want the output to appear, i.e. the run directory.
# E.g., if you cloned spectre in your home directory, set
# SPECTRE_BUILD_DIR to ${HOME}/spectre/build. If you want to run in a
# directory called "Run" in the current directory, set
# SPECTRE_RUN_DIR to ${PWD}/Run
# On Frontera, you'll most likely have cloned spectre to scratch, the $HOME
# option won't work, so make sure to write out full path from scratch.
export SPECTRE_BUILD_DIR=${HOME}/Codes/spectre/spectre/build_clang
export SPECTRE_MODULE_DIR=${HOME}/Codes/spectre_deps/modules/
export SPECTRE_RUN_DIR=${PWD}/Run

# Choose the executable and input file to run
# To use an input file in the current directory, set
# SPECTRE_INPUT_FILE to ${PWD}/InputFileName.yaml
export SPECTRE_EXECUTABLE=${SPECTRE_BUILD_DIR}/bin/EvolveGhKerrSchild
export SPECTRE_INPUT_FILE=${PWD}/KerrTest

# These commands load the relevant modules and cd into the run directory,
# creating it if it doesn't exist
source ${SPECTRE_BUILD_DIR}/../support/Environments/frontera_gcc.sh
module use ${SPECTRE_MODULE_DIR}
spectre_load_modules
module list

mkdir -p ${SPECTRE_RUN_DIR}
cd ${SPECTRE_RUN_DIR}

# Copy the input file into the run directory, to preserve it
cp ${SPECTRE_INPUT_FILE} ${SPECTRE_RUN_DIR}/

# Print out diagnostic info
module list 2>&1
pwd
date

export IBRUN_TASKS_PER_NODE=1

checkpoints=0
current_checkpoint=000000
if [ -f ${SPECTRE_EXECUTABLE} ]; then
    if [ -f ${SPECTRE_INPUT_FILE} ]; then
        if [[ $checkpoints == 0 ]]; then
            ibrun -n ${SLURM_JOB_NUM_NODES} \
            ${SPECTRE_EXECUTABLE} ++ppn 55 \
            --input-file ${SPECTRE_INPUT_FILE} 2>&1
            sleep 10s
            # If a checkpoint is found add one to checkpoint and sumbit next job
            if test -e "${PWD}/SpectreCheckpoint$current_checkpoint"; then
                cp ../Frontera.sh .
                sed -i "s/^checkpoints=0/checkpoints=1/" Frontera.sh
                ssh login1.frontera.tacc.utexas.edu "cd $SPECTRE_RUN_DIR
                sbatch Frontera.sh"
            fi
        # Section to start from checkpoints
        elif [[ $checkpoints -gt 0 && checkpoints -lt 1000000 ]]; then
            ln -s ${PWD}/../SpectreCheckpoint$current_checkpoint .
            ibrun -n ${SLURM_JOB_NUM_NODES} ${SPECTRE_EXECUTABLE} ++ppn 55 \
            +restart SpectreCheckpoint$current_checkpoint --input-file \
            ${SPECTRE_INPUT_FILE} 2>&1
            sleep 10s
            # If next checkpoint was created modify variables/submit next job
            printf -v next_checkpoint %06d $checkpoints
            if test -e "${PWD}/SpectreCheckpoint$next_checkpoint"; then
                cp ../Frontera.sh .
                next_num_of_checkpoints=$(($checkpoints + 1))
                #Updating variables for the next possible checkpoint
                sed -i "s/^checkpoints=$checkpoints/"\
"checkpoints=$next_num_of_checkpoints/" Frontera.sh
    sed -i "s/^current_checkpoint=$current_checkpoint/"\
"current_checkpoint=$next_checkpoint/" Frontera.sh
                ssh login1.frontera.tacc.utexas.edu "cd $SPECTRE_RUN_DIR
                sbatch Frontera.sh"
            fi
        fi
    else
        echo "Could not find input file ${SPECTRE_INPUT_FILE}"
        exit 1
    fi
else
    echo "Could not find executable ${SPECTRE_EXECUTABLE}"
    exit 1
fi
