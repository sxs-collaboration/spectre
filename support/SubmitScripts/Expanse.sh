#!/bin/bash -
#SBATCH -o spectre.stdout
#SBATCH -e spectre.stderr
#SBATCH --ntasks-per-node 128
#SBATCH --mem=248G
#SBATCH -J KerrSchild
#SBATCH --nodes 2
#SBATCH -A csd275
#SBATCH -p compute
#SBATCH -t 0:03:00
#SBATCH -D .

# Distributed under the MIT License.
# See LICENSE.txt for details.

# To run a job on Expanse:
# - Set the -J, --nodes, and -t options above, which correspond to job name,
#   number of nodes, and wall time limit in HH:MM:SS, respectively.
# - Set an `#SBATCH -A` argument for the computing account id.
# - Set the build directory, run directory, executable name,
#   and input file below.
# - For large jobs (multiple nodes), use -p compute and --ntasks-per-node 128
# - For small jobs (fewer than 128 procs), use -p shared set the number of
#   processors as --ntasks-per-node
#
# NOTE: The executable will not be copied from the build directory, so if you
#       update your build directory this file will use the updated executable.
#
# Optionally, if you need more control over how SpECTRE is launched on
# Expanse you can edit the launch command at the end of this file directly.
#
# To submit the script to the queue run:
#   sbatch Expanse.sh

# Replace these paths with the path to your build directory, to the source root
# directory, the spectre dependencies module directory, and to the directory
# where you want the output to appear, i.e. the run directory.
# E.g., if you cloned spectre in your home directory, set
# SPECTRE_BUILD_DIR to ${HOME}/spectre/build. If you want to run in a
# directory called "Run" in the current directory, set
# SPECTRE_RUN_DIR to ${PWD}/Run
export SPECTRE_BUILD_DIR=${HOME}/spectre-build/
export SPECTRE_HOME=${HOME}/spectre/
export SPECTRE_MODULE_DIR=${HOME}/Codes/spectre_deps/modules/
export SPECTRE_RUN_DIR=${PWD}/Run

# Choose the executable and input file to run
# To use an input file in the current directory, set
# SPECTRE_INPUT_FILE to ${PWD}/InputFileName.yaml
export SPECTRE_EXECUTABLE=${SPECTRE_BUILD_DIR}/bin/EvolveGhKerrSchild
export SPECTRE_INPUT_FILE=${PWD}/KerrSchild.yaml

# These commands load the relevant modules and cd into the run directory,
# creating it if it doesn't exist
source ${SPECTRE_HOME}/support/Environments/expanse_gcc.sh
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

# Generate the nodefile
echo "Running on the following nodes:"
echo ${SLURM_NODELIST}
touch nodelist.$SLURM_JOBID
for node in $(echo $SLURM_NODELIST | scontrol show hostnames); do
  echo "host ${node}" >> nodelist.$SLURM_JOBID
done

# tuned for 4 communication threads -- initial tests indicated
# that was most efficient for a basic KerrSchild executable with
# around 2k elements on 2 nodes.
# to adjust the number of comm threads:
#  - set the denominator in WORKER_THREADS_PER_NODE to number of comm threads
#  - set the multiplier of SLURM_NNODES in WORKER_THREADS to number of
#    comm threads
#  - adjust the pemap to be a number of ranges equal to the number of comm
#    threads, and appropriate intervals of 128/number_of_comm_threads - 1
#  - adjust the commap to be a number of values equal to the number of comm
#    threads, likely just n * 128/number_of_comm_theads, for
#    n=0..number_of_comm_threads
WORKER_THREADS_PER_NODE=$((SLURM_NTASKS_PER_NODE / 4 - 1))
WORKER_THREADS=$((SLURM_NPROCS - 4 * SLURM_NNODES))
SPECTRE_COMMAND="${SPECTRE_EXECUTABLE} \
++p ${WORKER_THREADS} ++ppn ${WORKER_THREADS_PER_NODE} \
+pemap 1-31,33-63,65-95,97-127 +commap 0,32,64,96 \
++nodelist nodelist.${SLURM_JOBID} +balancer RecBipartLB"

# When invoking through `charmrun`, charm will initiate remote sessions which
# will wipe out environment settings unless it is forced to re-initialize the
# spectre environment between the start of the remote session and starting the
# spectre executable
echo "#!/bin/sh
source ${SPECTRE_HOME}/support/Environments/expanse_gcc.sh
module use ${SPECTRE_MODULE_DIR}
spectre_load_modules
\$@
" > runscript


chmod u+x ./runscript

charmrun ++runscript ./runscript ${SPECTRE_COMMAND} \
         --input-file ${SPECTRE_INPUT_FILE}
