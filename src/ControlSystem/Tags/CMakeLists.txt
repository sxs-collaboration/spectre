# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  FunctionsOfTimeInitialize.cpp
  SystemTags.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  FunctionsOfTimeInitialize.hpp
  IsActive.hpp
  MeasurementTimescales.hpp
  OptionTags.hpp
  QueueTags.hpp
  SystemTags.hpp
  )
