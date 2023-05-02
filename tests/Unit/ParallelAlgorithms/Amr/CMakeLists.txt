# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ParallelAmr")

set(LIBRARY_SOURCES
  Actions/Test_AdjustDomain.cpp
  Actions/Test_CollectDataFromChildren.cpp
  Actions/Test_CreateChild.cpp
  Actions/Test_CreateParent.cpp
  Actions/Test_EvaluateRefinementCriteria.cpp
  Actions/Test_Initialize.cpp
  Actions/Test_InitializeChild.cpp
  Actions/Test_InitializeParent.cpp
  Actions/Test_SendDataToChildren.cpp
  Actions/Test_UpdateAmrDecision.cpp
  Criteria/Test_Criterion.cpp
  Criteria/Test_DriveToTarget.cpp
  Criteria/Test_Random.cpp
  Projectors/Test_Mesh.cpp
  )

add_test_library(
  ${LIBRARY}
  "ParallelAlgorithms/Amr"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Amr
  AmrCriteria
  AmrProjectors
  DomainStructure
  Utilities
  )
