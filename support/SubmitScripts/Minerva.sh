#!/bin/bash -
#SBATCH -J spectre
#SBATCH --nodes 2
#SBATCH -t 24:00:00
#SBATCH -p nr
#SBATCH -o spectre.out
#SBATCH -e spectre.out
#SBATCH --ntasks-per-node 16
#SBATCH --no-requeue

# Distributed under the MIT License.
# See LICENSE.txt for details.

# To run a job on Minerva:
#
# 1. Set the -J, --nodes, and -t options above, which correspond to job name,
#    number of nodes, and wall time limit in HH:MM:SS, respectively.
# 2. Set the build directory, run directory, executable name,
#    and input file below.
#    Note: The executable will not be copied from the build directory, so if you
#    update your build directory this file will use the updated executable.
# 3. Optionally, if you need more control over how SpECTRE is launched on
#    Minerva you can edit the launch command at the end of this file directly.
# 4. Submit the job to the queue:
#    ```sh
#    sbatch Minerva.sh
#    ```

# Replace these paths with the path to your build directory and to the
# directory where you want the output to appear, i.e. the run directory
export SPECTRE_BUILD_DIR=/work/nfischer/spectre/build_2021-03-18-Release
export SPECTRE_RUN_DIR=${PWD}

# Choose the executable and input file to run
# To use an input file in the current directory, set
# SPECTRE_INPUT_FILE to `${PWD}/InputFileName.yaml` or just `InputFileName.yaml`
export SPECTRE_EXECUTABLE=SolveXcts
export SPECTRE_INPUT_FILE=Schwarzschild.yaml

# --- You probably don't need to edit anything below this line ---

mkdir -p ${SPECTRE_RUN_DIR} && cd ${SPECTRE_RUN_DIR}

# Copy the input file into the run directory to preserve it
cp ${SPECTRE_INPUT_FILE} ${SPECTRE_RUN_DIR}/

# Set up the environment
export MODULEPATH="\
/home/SPACK2021/share/spack/modules/linux-centos7-haswell:$MODULEPATH"
export MODULEPATH="\
/home/nfischer/spack/share/spack/modules/linux-centos7-haswell:$MODULEPATH"
module purge
module load gcc-10.2.0-gcc-10.2.0-vaerku7
module load binutils-2.36.1-gcc-10.2.0-wtzd7wm
source /home/nfischer/spack/var/spack/environments/spectre_2021-03-18/loads

# Set permissions for files created with this script
umask 0022

# Add SpECTRE executables to the path
export PATH=${SPECTRE_BUILD_DIR}/bin:$PATH

# Run the executable on the available nodes. Notes:
# - We place only one proc per node because we're running with SMP.
# - We run on 15 procs per 16-core-node because Charm++ uses one thread per node
#   for communication.
# - IntelMPI, OpenMPI, srun and charmrun have different interfaces for similar
#   things. IntelMPI uses `-n NUM_NODES` and `-ppn PROCS_PER_NODE`, and OpenMPI
#   uses `-np NUM_NODES` and `--map-by ppr:PROCS_PER_NODE:node`. We are
#   currently using IntelMPI on Minerva.
mpirun -n ${SLURM_JOB_NUM_NODES} -ppn 1 \
  ${SPECTRE_EXECUTABLE} +ppn 15 +pemap 0-14 +commap 15 \
    --input-file ${SPECTRE_INPUT_FILE}
