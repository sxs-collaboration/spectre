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
