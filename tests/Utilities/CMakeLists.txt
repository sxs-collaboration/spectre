# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "TestUtilities")

set(LIBRARY_SOURCES
  RandomUnitNormal.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})
