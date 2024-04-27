{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Public supercomputer at Caltech using the icelake intel nodes.
# More information:
# https://www.hpc.caltech.edu/documentation

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 32
#SBATCH -p {{ queue | default("expansion") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
#SBATCH --constraint=icelake
{% endblock %}
