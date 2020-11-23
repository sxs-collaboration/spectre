# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  Cce
  PRIVATE
  InitializeJ.cpp
  InverseCubic.cpp
  NoIncomingRadiation.cpp
  ZeroNonSmooth.cpp
  )

spectre_target_headers(
  Cce
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  InitializeJ.hpp
  InverseCubic.hpp
  NoIncomingRadiation.hpp
  RegisterInitializeJWithCharm.hpp
  ZeroNonSmooth.hpp
  )
