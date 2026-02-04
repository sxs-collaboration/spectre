# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.IO.Exporter.Python.InterpolateToPoints"
  Test_InterpolateToPoints.py
  "unit;python"
  PyExporter)

spectre_add_python_bindings_test(
  "Unit.IO.Exporter.Python.SpacetimeInterpolator"
  Test_SpacetimeInterpolator.py
  "unit;python"
  PyExporter)

if(Boost_VERSION VERSION_GREATER_EQUAL 1.81.0)
  spectre_add_python_bindings_test(
    "Unit.IO.Exporter.Python.ModalSpacetimeInterpolator"
    Test_ModalSpacetimeInterpolator.py
    "unit;python"
    PyExporter)
endif()
