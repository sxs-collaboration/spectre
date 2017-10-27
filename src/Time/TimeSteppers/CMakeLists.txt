# Distributed under the MIT License.
# See LICENSE.txt for details.

set(MY_LIBRARY_SOURCES
  TimeSteppers/AdamsBashforthN.cpp
  TimeSteppers/RungeKutta3.cpp
  )

set(LIBRARY_SOURCES "${LIBRARY_SOURCES};${MY_LIBRARY_SOURCES}" PARENT_SCOPE)
