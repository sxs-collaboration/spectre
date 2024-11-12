# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_PostNewtonianHelpers")

set(LIBRARY_SOURCES
  Test_BinaryTrajectories.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  PostNewtonianHelpers
  )

spectre_add_python_bindings_test(
  "Unit.PostNewtonian.Python.BinaryTrajectories"
  "Test_BinaryTrajectories.py"
  "unit;Python"
  PyPostNewtonianHelpers
  )
