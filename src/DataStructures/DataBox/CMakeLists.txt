# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  DataStructures
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  DataBox.hpp
  DataBoxTag.hpp
  DataOnSlice.hpp
  Deferred.hpp
  Item.hpp
  PrefixHelpers.hpp
  Prefixes.hpp
  SubitemTag.hpp
  Subitems.hpp
  Tag.hpp
  TagName.hpp
  TagTraits.hpp
  )
