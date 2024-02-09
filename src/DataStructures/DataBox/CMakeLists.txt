# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Access.hpp
  AsAccess.hpp
  DataBox.hpp
  DataBoxTag.hpp
  DataOnSlice.hpp
  IsApplyCallable.hpp
  Item.hpp
  ObservationBox.hpp
  PrefixHelpers.hpp
  Prefixes.hpp
  SubitemTag.hpp
  Subitems.hpp
  Tag.hpp
  TagName.hpp
  TagTraits.hpp
  ValidateSelection.hpp
  )

add_subdirectory(Protocols)
