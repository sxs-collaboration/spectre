# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Brick"
  Test_Brick.py
  "Unit;Domain"
  PyDomainCreators)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Cylinder"
  Test_Cylinder.py
  "Unit;Domain"
  PyDomainCreators)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Interval"
  Test_Interval.py
  "Unit;Domain"
  PyDomainCreators)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Rectangle"
  Test_Rectangle.py
  "Unit;Domain"
  PyDomainCreators)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Shell"
  Test_Shell.py
  "Unit;Domain"
  PyDomainCreators)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.Python.Sphere"
  Test_Sphere.py
  "Unit;Domain"
  PyDomainCreators)
