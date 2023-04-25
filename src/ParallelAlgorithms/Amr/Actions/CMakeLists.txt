# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Actions.hpp
  AdjustDomain.hpp
  CollectDataFromChildren.hpp
  Component.hpp
  CreateChild.hpp
  CreateParent.hpp
  EvaluateRefinementCriteria.hpp
  Initialization.hpp
  Initialize.hpp
  InitializeChild.hpp
  InitializeParent.hpp
  RunAmrDiagnostics.hpp
  SendAmrDiagnostics.hpp
  SendDataToChildren.hpp
  UpdateAmrDecision.hpp
  )
