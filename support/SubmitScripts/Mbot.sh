{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Mbot is a supercomputer at Cornell hosted by Nils Deppe.
# More information:
# https://github.com/sxs-collaboration/WelcomeToSXS/wiki/Mbot

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node 6
#SBATCH --cpus-per-task 32
#SBATCH -p {{ queue | default("normal") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
{% endblock %}
