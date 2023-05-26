# Distributed under the MIT License.
# See LICENSE.txt for details.

# Optionally link SpEC libraries. Pass `SPEC_ROOT` to the CMake build
# configuration to set up the following targets:
#
# - SpEC::Exporter: Functionality to load SpEC volume data and interpolate to
#   arbitrary points.

option(SPEC_ROOT "Path to the git directory of SpEC." OFF)

if (DEFINED ENV{SPEC_ROOT} AND NOT SPEC_ROOT)
  set(SPEC_ROOT "$ENV{SPEC_ROOT}")
endif()

if(NOT SPEC_ROOT)
  return()
endif()

message(STATUS "Linking with SpEC: ${SPEC_ROOT}")

# SpEC needs MPI.
# NOTE: You should use the same MPI as SpEC. At least the same distribution. So
# mixing OpenMPI and MPICH would be bad.
find_package(MPI COMPONENTS C)

add_library(SpEC::Exporter INTERFACE IMPORTED)
set(SPEC_EXPORTER_ROOT ${SPEC_ROOT}/Support/ApplyObservers/Exporter)
target_include_directories(SpEC::Exporter INTERFACE ${SPEC_EXPORTER_ROOT})
add_interface_lib_headers(
  TARGET SpEC::Exporter
  HEADERS
  Exporter.hpp
)
target_link_libraries(
  SpEC::Exporter
  INTERFACE
  MPI::MPI_C
  # The order of these next two lines is important
  ${SPEC_EXPORTER_ROOT}/ExporterFactoryObjects.o
  ${SPEC_EXPORTER_ROOT}/libPackagedExporter.a
)
