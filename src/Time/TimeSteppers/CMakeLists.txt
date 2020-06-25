# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  Time
  PRIVATE
  AdamsBashforthN.cpp
  DormandPrince5.cpp
  RungeKutta3.cpp
  RungeKutta4.cpp
  )

spectre_target_headers(
  Time
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdamsBashforthN.hpp
  DormandPrince5.hpp
  RungeKutta3.hpp
  RungeKutta4.hpp
  TimeStepper.hpp
  )
