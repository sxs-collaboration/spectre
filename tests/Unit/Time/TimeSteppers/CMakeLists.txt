# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY_SOURCES
  ${LIBRARY_SOURCES}
  TimeSteppers/Test_AdamsBashforth.cpp
  TimeSteppers/Test_AdamsCoefficients.cpp
  TimeSteppers/Test_AdamsMoultonPc.cpp
  TimeSteppers/Test_ClassicalRungeKutta4.cpp
  TimeSteppers/Test_DormandPrince5.cpp
  TimeSteppers/Test_Heun2.cpp
  TimeSteppers/Test_Rk3HesthavenSsp.cpp
  TimeSteppers/Test_Rk3Owren.cpp
  TimeSteppers/Test_Rk4Owren.cpp
  TimeSteppers/Test_Rk5Owren.cpp
  TimeSteppers/Test_Rk5Tsitouras.cpp
  PARENT_SCOPE)
