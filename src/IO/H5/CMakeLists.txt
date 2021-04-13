# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AccessType.cpp
  Dat.cpp
  File.cpp
  Header.cpp
  Helpers.cpp
  OpenGroup.cpp
  SourceArchive.cpp
  SpectralIo.cpp
  StellarCollapseEos.cpp
  Version.cpp
  VolumeData.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AccessType.hpp
  CheckH5.hpp
  Dat.hpp
  File.hpp
  Header.hpp
  Helpers.hpp
  Object.hpp
  OpenGroup.hpp
  SourceArchive.hpp
  SpectralIo.hpp
  StellarCollapseEos.hpp
  Type.hpp
  Version.hpp
  VolumeData.hpp
  Wrappers.hpp
  )

add_subdirectory(Python)
