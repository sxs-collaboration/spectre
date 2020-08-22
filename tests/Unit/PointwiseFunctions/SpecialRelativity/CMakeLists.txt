# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_SpecialRelativity")

set(LIBRARY_SOURCES
  Test_LorentzBoostMatrix.cpp
  )

add_test_library(
  ${LIBRARY}
  "PointwiseFunctions/SpecialRelativity/"
  "${LIBRARY_SOURCES}"
  "SpecialRelativity"
  )
