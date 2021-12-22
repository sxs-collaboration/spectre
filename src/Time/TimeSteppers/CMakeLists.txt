# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AdamsBashforthN.cpp
  Cerk2.cpp
  Cerk3.cpp
  Cerk4.cpp
  DormandPrince5.cpp
  RungeKutta3.cpp
  RungeKutta4.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdamsBashforthN.hpp
  Cerk2.hpp
  Cerk3.hpp
  Cerk4.hpp
  DormandPrince5.hpp
  RungeKutta3.hpp
  RungeKutta4.hpp
  TimeStepper.hpp
  )
