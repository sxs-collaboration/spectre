# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY H5)

add_spectre_library(${LIBRARY})

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AccessType.cpp
  Cce.cpp
  CheckH5PropertiesMatch.cpp
  CombineH5.cpp
  Dat.cpp
  EosTable.cpp
  ExtendConnectivityHelpers.cpp
  File.cpp
  Header.cpp
  Helpers.cpp
  OpenGroup.cpp
  SourceArchive.cpp
  SpectralIo.cpp
  StellarCollapseEos.cpp
  TensorData.cpp
  Version.cpp
  VolumeData.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AccessType.hpp
  Cce.hpp
  CheckH5.hpp
  CheckH5PropertiesMatch.hpp
  CombineH5.hpp
  Dat.hpp
  EosTable.hpp
  ExtendConnectivityHelpers.hpp
  File.hpp
  Header.hpp
  Helpers.hpp
  Object.hpp
  OpenGroup.hpp
  SourceArchive.hpp
  SpectralIo.hpp
  StellarCollapseEos.hpp
  TensorData.hpp
  Type.hpp
  Version.hpp
  VolumeData.hpp
  Wrappers.hpp
  )

target_link_libraries(
  ${LIBRARY}
  PUBLIC
  Boost::boost
  DataStructures
  Domain
  DomainStructure
  ErrorHandling
  HDF5::HDF5
  Serialization
  Spectral
  Utilities
  PRIVATE
  Informer
  IO
  Printf
  )

add_subdirectory(Python)
