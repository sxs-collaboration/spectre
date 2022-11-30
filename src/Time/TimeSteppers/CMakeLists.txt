# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AdamsBashforth.cpp
  Cerk3.cpp
  Cerk4.cpp
  Cerk5.cpp
  ClassicalRungeKutta4.cpp
  DormandPrince5.cpp
  Heun2.cpp
  RungeKutta.cpp
  RungeKutta3.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdamsBashforth.hpp
  Cerk3.hpp
  Cerk4.hpp
  Cerk5.hpp
  ClassicalRungeKutta4.hpp
  DormandPrince5.hpp
  Factory.hpp
  Heun2.hpp
  LtsTimeStepper.hpp
  RungeKutta.hpp
  RungeKutta3.hpp
  TimeStepper.hpp
  )
