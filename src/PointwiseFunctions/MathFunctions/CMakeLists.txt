# Distributed under the MIT License.
# See LICENSE.txt for details.

set(MY_LIBRARY_SOURCES
  MathFunctions/Gaussian.cpp
  MathFunctions/PowX.cpp
  )

set(LIBRARY_SOURCES "${LIBRARY_SOURCES};${MY_LIBRARY_SOURCES}" PARENT_SCOPE)
