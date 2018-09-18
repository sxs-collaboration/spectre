# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY MathFunctions)

set(LIBRARY_SOURCES
  Gaussian.cpp
  PowX.cpp
  Sinusoid.cpp
  TensorProduct.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})

target_link_libraries(
  ${LIBRARY}
  INTERFACE DataStructures
  INTERFACE ErrorHandling
  )
