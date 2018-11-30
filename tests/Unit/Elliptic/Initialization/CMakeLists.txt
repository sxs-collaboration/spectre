# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EllipticInitialization")

set(LIBRARY_SOURCES
  Test_Domain.cpp
  )

add_test_library(
  ${LIBRARY}
  "Elliptic/Initialization/"
  "${LIBRARY_SOURCES}"
  "DataStructures;Domain;ErrorHandling;LinearOperators;Spectral"
  )
