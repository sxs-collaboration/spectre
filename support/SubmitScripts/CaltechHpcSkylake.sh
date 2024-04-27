{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# CaltechHPC is a supercomputer at Caltech using the skylake intel nodes.
# More information:
# https://www.hpc.caltech.edu/documentation

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 28
#SBATCH -p {{ queue | default("expansion") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
#SBATCH --reservation sxs
{% endblock %}
