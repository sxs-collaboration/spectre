.. Distributed under the MIT License.
   See LICENSE.txt for details.

The SpECTRE Python interface
============================

The SpECTRE Python interface exposes some SpECTRE functionality in Python. For
example, it helps with visualizing data, scheduling simulations, and working
with simulation output. To get started quickly, compile the ``all-pybindings``
(or ``cli``) target and then try running the command-line interface (CLI):

.. code-block:: sh

   # In the build directory:
   ./bin/spectre --help

Or jump into a Jupyter notebook where you can import the ``spectre`` modules:

.. code-block:: sh

   # In the build directory:
   ./bin/python-spectre -m jupyter notebook


Documentation
-------------

.. toctree::
   :maxdepth: 2

   Getting started <https://spectre-code.org/spectre_using_python.html>
   C++ documentation <https://spectre-code.org>


Python modules
--------------

.. autosummary::
   :toctree: _autosummary
   :recursive:
   :caption: Python modules

   spectre
