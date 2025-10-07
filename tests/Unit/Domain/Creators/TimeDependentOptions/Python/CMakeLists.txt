# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.TimeDependentOptions.Python.BinaryCompactObject"
  Test_BinaryCompactObject.py
  "Unit;Domain"
  PyDomainTimeDependentOptions
)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.TimeDependentOptions.Python.ExpansionMap"
  Test_ExpansionMap.py
  "Unit;Domain"
  PyDomainTimeDependentOptions
)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.TimeDependentOptions.Python.GridCenters"
  Test_GridCenters.py
  "Unit;Domain"
  PyDomainTimeDependentOptions
)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.TimeDependentOptions.Python.RotationMap"
  Test_RotationMap.py
  "Unit;Domain"
  PyDomainTimeDependentOptions
)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.TimeDependentOptions.Python.ShapeMap"
  Test_ShapeMap.py
  "Unit;Domain"
  PyDomainTimeDependentOptions
)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.TimeDependentOptions.Python.SkewMap"
  Test_SkewMap.py
  "Unit;Domain"
  PyDomainTimeDependentOptions
)

spectre_add_python_bindings_test(
  "Unit.Domain.Creators.TimeDependentOptions.Python.TranslationMap"
  Test_TranslationMap.py
  "Unit;Domain"
  PyDomainTimeDependentOptions
)
