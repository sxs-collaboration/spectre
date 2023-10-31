# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_Tensor")

set(LIBRARY_SOURCES
  Test_ContractFirstNIndices.cpp
  Test_Identity.cpp
  Test_Metafunctions.cpp
  Test_Slice.cpp
  Test_Tensor.cpp
  )

add_subdirectory(EagerMath)
add_subdirectory(Expressions)
add_subdirectory(Python)

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  Boost::boost
  DataStructures
  DataStructuresHelpers
  DomainStructure
  ErrorHandling
  Utilities
  Spectral
  )
