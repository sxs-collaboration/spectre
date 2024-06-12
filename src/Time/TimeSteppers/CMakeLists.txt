# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  AdamsBashforth.cpp
  AdamsCoefficients.cpp
  AdamsLts.cpp
  AdamsMoultonPc.cpp
  ClassicalRungeKutta4.cpp
  DormandPrince5.cpp
  Heun2.cpp
  ImexRungeKutta.cpp
  Rk3HesthavenSsp.cpp
  Rk3Kennedy.cpp
  Rk3Owren.cpp
  Rk3Pareschi.cpp
  Rk4Kennedy.cpp
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
  AdamsLts.hpp
  AdamsMoultonPc.hpp
  ClassicalRungeKutta4.hpp
  DormandPrince5.hpp
  Factory.hpp
  Heun2.hpp
  ImexRungeKutta.hpp
  ImexTimeStepper.hpp
  LtsTimeStepper.hpp
  Rk3HesthavenSsp.hpp
  Rk3Kennedy.hpp
  Rk3Owren.hpp
  Rk4Kennedy.hpp
  Rk4Owren.hpp
  Rk5Owren.hpp
  Rk3Pareschi.hpp
  Rk5Tsitouras.hpp
  RungeKutta.hpp
  TimeStepper.hpp
  )
