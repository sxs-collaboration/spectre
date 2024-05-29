# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdvanceTime.hpp
  ChangeStepSize.hpp
  CleanHistory.hpp
  RecordTimeStepperData.hpp
  SelfStartActions.hpp
  UpdateU.hpp
  )
