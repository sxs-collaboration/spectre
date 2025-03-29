{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Ocean2 is a supercomputer at Cal State Fullerton hosted by Geoffrey Lovelace.
# More information:
# Ask Geoffrey Lovelace for more information on Ocean2.

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node 6
#SBATCH --cpus-per-task 32
#SBATCH -p {{ queue | default("normal") }}
#SBATCH -t {{ time_limit | default("01-00:00:00") }}
{% endblock %}

{% block run_command %}
export OPENBLAS_NUM_THREADS=1

# Generate nodelist file
echo "Running on the following nodes:"
echo ${SLURM_NODELIST}
touch nodelist.$SLURM_JOBID
for node in $(echo $SLURM_NODELIST | scontrol show hostnames); do
  echo "host ${node}" >> nodelist.$SLURM_JOBID
done

# Set worker threads and run command
WORKER_THREADS=$((SLURM_NTASKS * CHARM_PPN))
SPECTRE_COMMAND="${SPECTRE_EXECUTABLE} ++np ${SLURM_NTASKS} \
++p ${WORKER_THREADS} ++ppn ${CHARM_PPN} \
++nodelist nodelist.${SLURM_JOBID}"


# When invoking through `charmrun`, charm will initiate remote sessions which
# will wipe out environment settings unless it is forced to re-initialize the
# spectre environment between the start of the remote session and starting the
# spectre executable
echo "#!/bin/sh
source ${SPECTRE_HOME}/support/Environments/ocean2.sh
spectre_load_modules
\$@
" > ${RUN_DIR}/runscript.${SLURM_JOBID}
chmod u+x ${RUN_DIR}/runscript.${SLURM_JOBID}

# Run
charmrun ++runscript ${RUN_DIR}/runscript.${SLURM_JOBID} \
          ${SPECTRE_COMMAND} --input-file ${SPECTRE_INPUT_FILE} \
          ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}

