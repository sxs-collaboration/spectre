{% extends "!autosummary/module.rst" %}

.. Distributed under the MIT License.
   See LICENSE.txt for details.

.. Include pybindings documentation. This won't do anything for modules that
   have no pybindings but print a warning.
{% block modules %}
.. automodule:: {{ fullname }}._Pybindings

.. currentmodule:: {{ fullname }}

{{ super() }}
{% endblock %}
