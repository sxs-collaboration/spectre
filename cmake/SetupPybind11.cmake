# Distributed under the MIT License.
# See LICENSE.txt for details.

option(BUILD_PYTHON_BINDINGS "Build the python bindings for SpECTRE" OFF)

if(BUILD_PYTHON_BINDINGS)
  # Make sure to find Python first so it's consistent with pybind11
  find_package(Python COMPONENTS Interpreter Development)

  # Try to find the pybind11-config tool to find pybind11's CMake config files
  find_program(PYBIND11_CONFIG_TOOL pybind11-config)
  set(PYBIND11_CMAKEDIR "")
  if(PYBIND11_CONFIG_TOOL)
    execute_process(
      COMMAND "${PYBIND11_CONFIG_TOOL}" "--cmakedir"
      OUTPUT_VARIABLE PYBIND11_CMAKEDIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Found pybind11-config tool (${PYBIND11_CONFIG_TOOL}) and "
      "determined CMake dir: ${PYBIND11_CMAKEDIR}")
  endif()

  find_package(pybind11 2.6.0 REQUIRED HINTS "${PYBIND11_CMAKEDIR}")

  set_property(
    GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
    pybind11::headers
    )

  message(STATUS "Pybind11 include: ${pybind11_INCLUDE_DIR}")
endif()
