# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ArrayIndex.hpp
  DistributedObjectTags.hpp
  InputSource.hpp
  Metavariables.hpp
  Parallelization.hpp
  ResourceInfo.hpp
  Section.hpp
)
