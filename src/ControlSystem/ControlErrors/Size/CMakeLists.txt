# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Info.cpp
  AhSpeed.cpp
  DeltaR.cpp
  Error.cpp
  Initial.cpp
  RegisterDerivedWithCharm.cpp
  StateHistory.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Info.hpp
  State.hpp
  StateHistory.hpp
  AhSpeed.hpp
  DeltaR.hpp
  Error.hpp
  Initial.hpp
  RegisterDerivedWithCharm.hpp
  Update.hpp
  )
