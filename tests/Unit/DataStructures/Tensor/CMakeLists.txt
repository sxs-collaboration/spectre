# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Tensor")

set(LIBRARY_SOURCES
  Test_Identity.cpp
  Test_Slice.cpp
  Test_Tensor.cpp
  Test_TensorData.cpp
  )

add_subdirectory(EagerMath)
add_subdirectory(Expressions)

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor"
  "${LIBRARY_SOURCES}"
  "Boost::boost;DataStructures;ErrorHandling"
  )

spectre_add_python_bindings_test(
  "Unit.DataStructures.Tensor.Python.TensorData"
  Test_TensorData.py
  "Unit;DataStructures;Tensor;Python"
  PyDataStructures)
