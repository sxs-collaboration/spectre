# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  ChangeSlabSize.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdvanceTime.hpp
  ChangeSlabSize.hpp
  ChangeStepSize.hpp
  LimitTimeStepToExpirationTimes.hpp
  RecordTimeStepperData.hpp
  SelfStartActions.hpp
  UpdateU.hpp
  )
