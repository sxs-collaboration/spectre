# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  IO
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ObserverRegistration.hpp
  RegisterEvents.hpp
  RegisterSingleton.hpp
  RegisterWithObservers.hpp
  )
