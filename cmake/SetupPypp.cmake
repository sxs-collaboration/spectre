# Distributed under the MIT License.
# See LICENSE.txt for details.

# To run Python tests we need the development component (libs and include dirs).
# It provides the `Python::Python` imported target. We find the interpreter
# component as well to make sure the find is consistent with earlier finds that
# only looked for the interpreter, possibly guided by the Python_EXECUTABLE
# variable set by the user.
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

# Also check that SciPy is installed
include(FindPythonModule)
find_python_module(scipy TRUE)

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Python::NumPy Python::Python
  )
