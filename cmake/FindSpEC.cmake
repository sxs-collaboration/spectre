# Distributed under the MIT License.
# See LICENSE.txt for details.

# Optionally link SpEC libraries. Pass `SPEC_ROOT` to the CMake build
# configuration to set up the following targets:
#
# - SpEC::Exporter: Functionality to load SpEC volume data and interpolate to
#   arbitrary points.

if(NOT SPEC_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(SPEC_ROOT "")
  set(SPEC_ROOT $ENV{SPEC_ROOT})
endif()

if (SPEC_ROOT)
  set(SPEC_EXPORTER_ROOT ${SPEC_ROOT}/Support/ApplyObservers/Exporter)
else()
  set(SPEC_EXPORTER_ROOT "")
endif()

find_library(
  SPEC_PACKAGED_EXPORTER_LIB
  NAMES libPackagedExporter.a
  PATHS ${SPEC_EXPORTER_ROOT}
  NO_DEFAULT_PATHS
  )
find_file(
  SPEC_EXPORTER_FACTORY_OBJECTS
  NAMES ExporterFactoryObjects.o
  PATHS ${SPEC_EXPORTER_ROOT}
  NO_DEFAULT_PATHS
  )
find_path(
  SPEC_EXPORTER_INCLUDE_DIR
  NAMES Exporter.hpp
  PATHS ${SPEC_EXPORTER_ROOT}
  NO_DEFAULT_PATHS
  )

# SpEC needs MPI.
# NOTE: You should use the same MPI as SpEC. At least the same distribution. So
# mixing OpenMPI and MPICH would be bad.
find_package(MPI COMPONENTS C)

if (SPEC_PACKAGED_EXPORTER_LIB AND SPEC_EXPORTER_FACTORY_OBJECTS AND
    SPEC_EXPORTER_INCLUDE_DIR AND MPI_C_FOUND)
  add_library(SpEC::Exporter INTERFACE IMPORTED)
  target_include_directories(
    SpEC::Exporter INTERFACE ${SPEC_EXPORTER_INCLUDE_DIR})
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
    ${SPEC_EXPORTER_FACTORY_OBJECTS}
    ${SPEC_PACKAGED_EXPORTER_LIB}
  )

  # Deal with FFTW3
  #
  # If it was dynamically linked into SpEC then we need to dynamically link it
  # into SpECTRE.
  set(_MACHINE_DEF_FILE
    ${SPEC_ROOT}/MakefileRules/this_machine.def)
  if (EXISTS ${_MACHINE_DEF_FILE})
    file(STRINGS ${_MACHINE_DEF_FILE} _MACHINE_DEF)
    # Filter starting comments
    list(FILTER _MACHINE_DEF EXCLUDE REGEX "^[ ]*#")
    # Remove inline comments
    list(TRANSFORM _MACHINE_DEF REPLACE "[ ]*#.*" "")
    string(REGEX MATCH "FFTW_LIB[^;]+"
      _FFTW3_DEF "${_MACHINE_DEF}")
    string(FIND ${_FFTW3_DEF} "-lfftw3" _FOUND_FFTW3_SHARED)
    # If we found a shared FFTW lib in SpEC, find and link FFTW here
    if(NOT ${_FOUND_FFTW3_SHARED} STREQUAL "-1")
      find_package(FFTW REQUIRED)
      target_link_libraries(SpEC::Exporter INTERFACE FFTW::FFTW)
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  SpEC
  REQUIRED_VARS
  SPEC_PACKAGED_EXPORTER_LIB
  SPEC_EXPORTER_FACTORY_OBJECTS
  SPEC_EXPORTER_INCLUDE_DIR
  )
