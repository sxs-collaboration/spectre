# Distributed under the MIT License.
# See LICENSE.txt for details.

# Make sure to find the Python interpreter first, so it is consistent with
# the libs
find_package(PythonInterp)
find_package(PythonLibs REQUIRED)

message(STATUS "Python libs: " ${PYTHON_LIBRARIES})
message(STATUS "Python incl: " ${PYTHON_INCLUDE_DIRS})

add_library(PythonLibs INTERFACE IMPORTED)
set_property(TARGET PythonLibs PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${PYTHON_INCLUDE_DIRS})
set_property(TARGET PythonLibs PROPERTY
  INTERFACE_LINK_LIBRARIES ${PYTHON_LIBRARIES})
