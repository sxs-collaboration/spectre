# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  Time
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AdvanceTime.hpp
  ChangeSlabSize.hpp
  ChangeStepSize.hpp
  RecordTimeStepperData.hpp
  SelfStartActions.hpp
  UpdateU.hpp
  )
