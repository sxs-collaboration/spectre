# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})
list(APPEND SPECTRE_LIBRARIES ${PYTHON_LIBRARIES})

message(STATUS "Python libs: " ${PYTHON_LIBRARIES})
message(STATUS "Python incl: " ${PYTHON_INCLUDE_DIRS})
