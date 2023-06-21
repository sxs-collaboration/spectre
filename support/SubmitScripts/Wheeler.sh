{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Wheeler is a supercomputer at Caltech.
# More information:
# https://github.com/sxs-collaboration/WelcomeToSXS/wiki/Wheeler

{% block head %}
{{ super() -}}
#SBATCH --nodes {{ num_nodes | default(1) }}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 24
#SBATCH -A sxs
#SBATCH -p {{ queue | default("productionQ") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
{% endblock %}

# We had an issue calling 'module list' on the 'unlimited' queue
{% block list_modules %}
/usr/bin/modulecmd bash list
{% endblock %}
