#!/bin/bash
#SBATCH -J MyJobName       # Job name
#SBATCH -o spectre-%j.out  # Name of stdout output file
#SBATCH -e spectre-%j.err  # Name of stderr error file
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

# Replace these paths with the path where you cloned spectre,
# to the source root directory, the spectre dependencies module
# directory, and to the directory where you want the output to appear, i.e.
# the run directory. E.g., if you cloned spectre in your home directory, set
# SPECTRE_HOME to ${HOME}/spectre/. If you set up your module
# dependency in your home directory and named it "deps", set
# SPECTRE_MODULE_DIR=/${HOME}/deps/modules/. If you want to run in a directory
# called "Run" in the current directory, set SPECTRE_RUN_DIR to ${PWD}/Run
export SPECTRE_HOME=${HOME}/Codes/where/spectre/is/
export SPECTRE_MODULE_DIR=${HOME}/Codes/spectre_deps/modules/
export SPECTRE_RUN_DIR=${PWD}/Run

# Make sure SPECTRE_HOME is defined or else
# Choose the executable and input file to run
# To use an input file in the current directory, set
# SPECTRE_INPUT_FILE to ${PWD}/InputFileName.yaml
if [[ ${SPECTRE_HOME} == "${HOME}/Codes/where/spectre/is/" ]]; then
    echo "SPECTRE_HOME is not defined, please define it above"
    exit 1
fi
export SPECTRE_EXECUTABLE_NAME=EvolveGhKerrSchild
export SPECTRE_INPUT_FILE_NAME=KerrSchildTest
export SPECTRE_EXECUTABLE=${SPECTRE_HOME}/build/bin/${SPECTRE_EXECUTABLE_NAME}
export SPECTRE_INPUT_FILE=${PWD}/${SPECTRE_INPUT_FILE_NAME}

# These commands load the relevant modules and cd into the run directory,
# creating it if it doesn't exist
if [[ ${SPECTRE_MODULE_DIR} == "${HOME}/Codes/spectre_deps/modules/" ]]; then
    echo "SPECTRE_MODULE_DIR is not defined, please make sure the definition
    above is correct"
    exit 1
fi
source ${SPECTRE_HOME}/support/Environments/frontera_gcc.sh
module use ${SPECTRE_MODULE_DIR}
spectre_load_modules
mkdir -p ${SPECTRE_RUN_DIR}
cd ${SPECTRE_RUN_DIR}

# Print out diagnostic info
module list 2>&1
pwd
date

export IBRUN_TASKS_PER_NODE=1

if [ -f ${SPECTRE_EXECUTABLE} ] || [ -f ${PWD}/../${SPECTRE_EXECUTABLE_NAME}];
    then
    if [ -f ${SPECTRE_INPUT_FILE} ] || [ -f ${SPECTRE_INPUT_FILE_NAME} ]; then
        # $1 Tracks number of checkpoints with a default argument that = 0
        if [[ ${1:-0} == 0 ]]; then
            cp $SPECTRE_EXECUTABLE .
            cp ${SPECTRE_INPUT_FILE} ${SPECTRE_RUN_DIR}/
            ibrun -n ${SLURM_JOB_NUM_NODES} \
            ${SPECTRE_EXECUTABLE_NAME} ++ppn 55 \
            --input-file ${SPECTRE_INPUT_FILE_NAME} 2>&1
            sleep 10s
            # If a checkpoint is found add one to checkpoint and sumbit next job
            if test -e "${PWD}/SpectreCheckpoint000000"; then
                cp ../Frontera.sh .
                ssh login1.frontera.tacc.utexas.edu "cd $SPECTRE_RUN_DIR ;
                sbatch Frontera.sh 1 000000"
            fi
        # Section to start from checkpoints (restart argument is used again)
        elif [[ $1 -gt 0 && $1 -lt 1000000 ]]; then
            cp ${PWD}/../${SPECTRE_EXECUTABLE_NAME} .
            cp ../${SPECTRE_INPUT_FILE_NAME} ${SPECTRE_RUN_DIR}/
            ln -s ${PWD}/../SpectreCheckpoint$2 .
            ibrun -n ${SLURM_JOB_NUM_NODES} ${SPECTRE_EXECUTABLE_NAME}
            ++ppn 55 +restart SpectreCheckpoint$2 --input-file \
            ${SPECTRE_INPUT_FILE_NAME} 2>&1
            sleep 10s
            # If next checkpoint was created modify variables/submit next job
            printf -v next_checkpoint %06d $1
            if test -e "${PWD}/SpectreCheckpoint$next_checkpoint"; then
                cp ../Frontera.sh .
                next_num_of_checkpoints=$(($1 + 1))
                # Updating variables for the next possible checkpoint
                ssh login1.frontera.tacc.utexas.edu "cd $SPECTRE_RUN_DIR
                sbatch Frontera.sh $next_num_of_checkpoints $next_checkpoint"
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
