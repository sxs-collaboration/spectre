{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Orca-1 Partition on the Ocean2 Supercomputer at Cal State Fullerton hosted by
# Geoffrey Lovelace
# More information:
# Ask Geoffrey Lovelace for more information on Ocean2.

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 20
#SBATCH -p {{ queue | default("orca-1") }}
#SBATCH -t {{ time_limit | default("01-00:00:00") }}
{% endblock %}
