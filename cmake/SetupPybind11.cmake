# Distributed under the MIT License.
# See LICENSE.txt for details.

option(BUILD_PYTHON_BINDINGS "Build the python bindings for SpECTRE" OFF)

if(BUILD_PYTHON_BINDINGS)
  # Pybind11's internal python library finding functions break CMake's
  # FindPythonLibs (which we use). Thus, we need to configure both the
  # interpreter and the libs before configuring pybind11.
  include(SpectreFindPython)
  spectre_find_python(REQUIRED COMPONENTS Development Interpreter)

  # Uses `Findpybind11.cmake` to find the headers. Since we can't rely on the
  # corresponding cmake files to be installed as well we bundle them in
  # `external/pybind11`.
  find_package(pybind11 REQUIRED)

  # Load the CMake files from `external/pybind11`
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/pybind11/tools")
  include(pybind11Tools)

  message(STATUS "Pybind11 include: ${pybind11_INCLUDE_DIR}")
endif()
