# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "CceHelpers")

set(LIBRARY_SOURCES
  CceComputationTestHelpers.cpp
  )

add_spectre_library(${LIBRARY} ${LIBRARY_SOURCES})
