# Distributed under the MIT License.
# See LICENSE.txt for details.

add_charm_module(AlgorithmArray)
add_charm_module(AlgorithmGroup)
add_charm_module(AlgorithmNodegroup)
add_charm_module(AlgorithmSingleton)

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  AlgorithmArray.hpp
  AlgorithmArrayDeclarations.hpp
  AlgorithmGroup.hpp
  AlgorithmGroupDeclarations.hpp
  AlgorithmNodegroup.hpp
  AlgorithmNodegroupDeclarations.hpp
  AlgorithmSingleton.hpp
  AlgorithmSingletonDeclarations.hpp
)

add_dependencies(
  ${LIBRARY}
  module_AlgorithmArray
  module_AlgorithmGroup
  module_AlgorithmNodegroup
  module_AlgorithmSingleton
  )
