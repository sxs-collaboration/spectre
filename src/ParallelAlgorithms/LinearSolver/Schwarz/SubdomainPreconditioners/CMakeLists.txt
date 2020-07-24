# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ParallelSchwarz
  PRIVATE
  ExplicitInverse.cpp
  )

spectre_target_headers(
  ParallelSchwarz
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ExplicitInverse.hpp
  )
