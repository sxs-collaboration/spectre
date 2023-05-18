#!/bin/bash -
{% block head %}
#SBATCH -o {{ out_file }}
#SBATCH -e {{ out_file }}
#SBATCH -J {{ job_name }}
#SBATCH --no-requeue
#SBATCH --comment "SPECTRE_INPUT_FILE={{ input_file }}"
{% endblock %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

export RUN_DIR={{ run_dir }}
export SPECTRE_INPUT_FILE={{ input_file }}
export SPECTRE_EXECUTABLE={{ executable }}
export SPECTRE_CHECKPOINT={{ from_checkpoint | default("") }}
export SPECTRE_CLI={{ spectre_cli }}

{% block charm_ppn %}
CHARM_PPN=$(expr ${SLURM_CPUS_PER_TASK} - 1)
{% endblock %}

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
echo "Executable: ${SPECTRE_EXECUTABLE}"
echo "CLI: ${SPECTRE_CLI}"
echo

{% block list_modules %}
module list
{% endblock %}

############################################################################
# Set desired permissions for files created with this script
umask 0022

echo
echo "###################################"
echo "######   Executable Output   ######"
echo "###################################"
echo

cd ${RUN_DIR}

{% block run_command %}
mpirun -n ${SLURM_NTASKS} \
  ${SPECTRE_EXECUTABLE} --input-file ${SPECTRE_INPUT_FILE} \
  ++ppn ${CHARM_PPN} +setcpuaffinity \
  ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}

{% if segments_dir is defined %}
# Resubmit
exit_code=$?
if [ $exit_code -eq 2 ]; then
  sleep 10s
  ${SPECTRE_CLI} resubmit . \
    --context-file-name {{ context_file_name }} \
    --submit
else
  exit $exit_code
fi
{% endif %}
