{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# CPU partition of the CSCS Alps system.
# More information:
# https://docs.cscs.ch/clusters/eiger/

# SpECTRE runs in the sxscollaboration/spectre container with Apptainer. Pass
# `-p spectre_image=/path/to/spectre.sif` and optionally `-p account=...` and
# `-p apptainer=/path/to/apptainer`. There is no Slurm inside the container, so
# segmented runs can't resubmit themselves.

{% block head %}
{{ super() -}}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --hint=nomultithread
{% if account is defined %}#SBATCH -A {{ account }}
{% endif -%}
#SBATCH -p {{ queue | default("normal") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
{% endblock %}

{% block charm_ppn %}
# Multicore Charm++: no communication thread
CHARM_PPN=${SLURM_CPUS_PER_TASK}
{% endblock %}

{% block list_modules %}
# Run the executable and the CLI in the container, with the build's bin
# directory on the PATH
SPECTRE_CONTAINER="{{ apptainer | default("apptainer") }} exec \
  --cleanenv \
  --env PREPEND_PATH=$(dirname "${SPECTRE_CLI}") \
  --bind {{ apptainer_binds | default("/capstor,/ritom") }} \
  {{ spectre_image }}"
export SPECTRE_CLI="${SPECTRE_CONTAINER} ${SPECTRE_CLI}"
{% endblock %}

{% block run_command %}
${SPECTRE_CONTAINER} \
  ${SPECTRE_PROFILING_PREFIX} \
  ${SPECTRE_EXECUTABLE} --input-file ${SPECTRE_INPUT_FILE} \
  +p${CHARM_PPN} +setcpuaffinity \
  ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}
