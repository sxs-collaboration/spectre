# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DataBox.hpp
  DataBoxTag.hpp
  DataOnSlice.hpp
  Item.hpp
  ObservationBox.hpp
  PrefixHelpers.hpp
  Prefixes.hpp
  SubitemTag.hpp
  Subitems.hpp
  Tag.hpp
  TagName.hpp
  TagTraits.hpp
  )
