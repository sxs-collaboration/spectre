# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY IO)

set(LIBRARY_SOURCES
  Connectivity.cpp
  H5/AccessType.cpp
  H5/Dat.cpp
  H5/File.cpp
  H5/Header.cpp
  H5/Helpers.cpp
  H5/OpenGroup.cpp
  H5/SourceArchive.cpp
  H5/StellarCollapseEos.cpp
  H5/Version.cpp
  H5/VolumeData.cpp
  Observer/ArrayComponentId.cpp
  Observer/ObservationId.cpp
  Observer/TypeOfObservation.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  INTERFACE Utilities
  )

if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(H5/Python)
endif()
