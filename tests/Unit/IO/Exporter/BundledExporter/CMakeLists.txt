# Distributed under the MIT License.
# See LICENSE.txt for details.

# This is a standalone CMake project to test the SpECTRE exporter library in an
# external project. It is not part of the SpECTRE build system.
#
# Instructions to build this project:
# - Configure a build directory and set the `SPECTRE_ROOT` variable to the
#   directory containing the compiled SpECTRE libraries, e.g. the build
#   directory of the SpECTRE project:
#
#   mkdir build && cd build
#   cmake -D SPECTRE_ROOT=/path/to/spectre/build/directory /path/to/this/project
#
# - Build and run the test executable:
#
#   make
#   ./TestSpectreExporter --help
#
#   The C interface is tested by the `TestSpectreExporterC` executable, which
#   takes the same arguments.
#
# - The Mathematica interface can be tested with `Test_BundledExporter.wls`
#   (requires a Wolfram Engine, so it doesn't run in CI):
#
#   wolframscript -file /path/to/this/project/Test_BundledExporter.wls \
#     /path/to/spectre/build/directory ARGS...

cmake_minimum_required(VERSION 3.18)
project(TestSpectreExporter LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# Find SpECTRE exporter lib
find_library(
  SPECTRE_EXPORTER_LIB
  NAMES BundledExporter
  PATH_SUFFIXES lib
  PATHS ${SPECTRE_ROOT})
find_path(
  SPECTRE_EXPORTER_INCLUDE_DIR
  NAMES spectre/Exporter.hpp
  PATH_SUFFIXES include
  PATHS ${SPECTRE_ROOT})
add_library(spectre::Exporter UNKNOWN IMPORTED)
set_target_properties(
  spectre::Exporter
  PROPERTIES IMPORTED_LOCATION ${SPECTRE_EXPORTER_LIB}
             INTERFACE_INCLUDE_DIRECTORIES ${SPECTRE_EXPORTER_INCLUDE_DIR})

# Find external libs
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(BLAS REQUIRED)

# Define test executables
add_executable(TestSpectreExporter Test_BundledExporter.cpp)
target_link_libraries(TestSpectreExporter PRIVATE spectre::Exporter)
add_executable(TestSpectreExporterC Test_BundledExporter.c)
target_link_libraries(TestSpectreExporterC PRIVATE spectre::Exporter)
