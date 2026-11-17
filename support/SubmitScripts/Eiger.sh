{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# CPU partition of the CSCS Alps system.
# More information:
# https://docs.cscs.ch/clusters/eiger/

# SpECTRE runs in the sxscollaboration/spectre container with the CSCS
# Container Engine. Pass `-p container_image=...` if SpECTRE was built in
# another image than `sxscollaboration/spectre:dev`, and optionally
# `-p account=...`.

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
# Write the container definition file for the CSCS Container Engine. It adds
# some host libraries to the container, which can conflict with the image's, so
# the image's libraries (in /usr/lib/x86_64-linux-gnu) go first.
cat > ${RUN_DIR}/container-env.toml <<'EOF'
image = "{{ container_image | default("sxscollaboration/spectre:dev") }}"
mounts = ["${HOME}:${HOME}", "${SCRATCH}:${SCRATCH}", "/capstor:/capstor"]
workdir = "${RUN_DIR}"
[env]
LD_LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu"
EOF
# Set a stack limit for the job steps (unlimited by default here) so that
# threads get enough stack
ulimit -s 8192
# Run the executable and the CLI in the container
SPECTRE_CONTAINER="srun --propagate=STACK \
  --environment=${RUN_DIR}/container-env.toml"
export SPECTRE_CLI="${SPECTRE_CONTAINER} ${SPECTRE_CLI}"
{% endblock %}

{% block run_command %}
${SPECTRE_CONTAINER} \
  ${SPECTRE_PROFILING_PREFIX} \
  ${SPECTRE_EXECUTABLE} --input-file ${SPECTRE_INPUT_FILE} \
  +p${CHARM_PPN} +setcpuaffinity \
  ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}
