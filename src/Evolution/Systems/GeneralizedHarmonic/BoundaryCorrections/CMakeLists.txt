# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AveragedUpwindPenalty.cpp
  UpwindPenalty.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AveragedUpwindPenalty.hpp
  Factory.hpp
  NamespaceDocs.hpp
  UpwindPenalty.hpp
  )
