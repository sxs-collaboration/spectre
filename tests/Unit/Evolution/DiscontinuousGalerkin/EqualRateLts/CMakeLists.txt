# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Tags)

set(LIBRARY_SOURCES
  ${LIBRARY_SOURCES}
  EqualRateLts/Test_EqualRateRegionGenerator.cpp
  EqualRateLts/Test_EqualRateRegions.cpp
  EqualRateLts/Test_NonconformingEqualRateRegions.cpp
  PARENT_SCOPE)
