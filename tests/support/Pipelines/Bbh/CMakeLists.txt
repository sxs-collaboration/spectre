# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.FindHorizon"
  Test_FindHorizon.py
  "Python"
  None
  TIMEOUT 20)

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.InitialData"
  Test_InitialData.py
  "Python"
  None
  TIMEOUT 60)

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.Inspiral"
  Test_Inspiral.py
  "Python"
  None
  TIMEOUT 60)

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.PostprocessId"
  Test_PostprocessId.py
  "Python"
  None
  TIMEOUT 60)

spectre_add_python_bindings_test(
  "support.Pipelines.Bbh.Ringdown"
  Test_Ringdown.py
  "Python"
  None
  TIMEOUT 60)
