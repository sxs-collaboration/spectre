#!/bin/bash
#SBATCH --job-name=spectre_tier1_inspiral
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=56
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=3984284@myuwc.ac.za

echo "====================================================="
echo "Starting SpECTRE Run: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Cores allocated: $SLURM_NTASKS"
echo "====================================================="

# Load required cluster modules (update these based on the cluster's specific MPI/compiler names when you get access)
module purge
module load gcc/11.2.0
module load openmpi/4.1.4
module load hdf5/1.12.1

# Set the path to the compiled executable and your input file
SPECTRE_EXE="../../build/bin/EvolveGhBinaryBlackHole"
INPUT_YAML="../tests/InputFiles/GeneralizedHarmonic/BinaryBlackHole_Inspiral.yaml"

# Create output directory for this specific run
mkdir -p run_output_${SLURM_JOB_ID}
cd run_output_${SLURM_JOB_ID}

# Execute SpECTRE via MPI
echo "Launching SpECTRE at $(date)"
mpirun -np $SLURM_NTASKS $SPECTRE_EXE --input-file $INPUT_YAML

echo "====================================================="
echo "Run completed at $(date)"
echo "====================================================="
