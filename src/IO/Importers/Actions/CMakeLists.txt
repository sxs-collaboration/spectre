# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  Importers
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ReadVolumeData.hpp
  ReceiveVolumeData.hpp
  RegisterWithElementDataReader.hpp
  )
