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
  Criteria/Test_Constraints.cpp
  Criteria/Test_Criterion.cpp
  Criteria/Test_DriveToTarget.cpp
  Criteria/Test_IncreaseResolution.cpp
  Criteria/Test_Loehner.cpp
  Criteria/Test_Persson.cpp
  Criteria/Test_Random.cpp
  Criteria/Test_TruncationError.cpp
  Events/Test_RefineMesh.cpp
  Policies/Test_EnforcePolicies.cpp
  Policies/Test_Isotropy.cpp
  Policies/Test_Limits.cpp
  Policies/Test_Policies.cpp
  Projectors/Test_CopyFromCreatorOrLeaveAsIs.cpp
  Projectors/Test_DefaultInitialize.cpp
  Projectors/Test_Mesh.cpp
  Projectors/Test_Tensors.cpp
  Projectors/Test_Variables.cpp
  Test_Protocols.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Amr
  AmrCriteria
  AmrEvents
  AmrPolicies
  AmrProjectors
  CoordinateMaps
  Domain
  DomainStructure
  Utilities
  )
