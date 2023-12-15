# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  ImplicitHistory.hpp
  Jacobian.hpp
  Mode.hpp
  NamespaceDocs.hpp
  OptionGroup.hpp
  SolveFailures.hpp
  SolveTolerance.hpp
  )
