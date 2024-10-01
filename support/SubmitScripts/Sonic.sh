{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Sonic is an HPC at ICTS-TIFR hosted by its AstroRel group.
# More information:
# https://it.icts.res.in/docs/sonic-cluster/

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node 3
#SBATCH --cpus-per-task 32
#SBATCH -p {{ queue | default("long") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
{% endblock %}
