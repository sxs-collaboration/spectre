{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Supercomputer at Brown.
# More information:
# https://docs.ccv.brown.edu/oscar

# This submit script currently requests cascadelake nodes (with 32 cores per
# node). We can adjust it to request other types of nodes later.

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 16
#SBATCH -p {{ queue | default("batch") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
#SBATCH --constraint=cascade
{% endblock %}
