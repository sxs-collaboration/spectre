# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AdamsBashforth.cpp
  AdamsCoefficients.cpp
  AdamsMoultonPc.cpp
  ClassicalRungeKutta4.cpp
  DormandPrince5.cpp
  Heun2.cpp
  Rk3HesthavenSsp.cpp
  Rk3Owren.cpp
  Rk4Owren.cpp
  Rk5Owren.cpp
  Rk5Tsitouras.cpp
  RungeKutta.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdamsBashforth.hpp
  AdamsCoefficients.hpp
  AdamsMoultonPc.hpp
  ClassicalRungeKutta4.hpp
  DormandPrince5.hpp
  Factory.hpp
  Heun2.hpp
  LtsTimeStepper.hpp
  Rk3HesthavenSsp.hpp
  Rk3Owren.hpp
  Rk4Owren.hpp
  Rk5Owren.hpp
  Rk5Tsitouras.hpp
  RungeKutta.hpp
  TimeStepper.hpp
  )
