# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EagerMath")

set(LIBRARY_SOURCES
  Test_CartesianToSpherical.cpp
  Test_CrossProduct.cpp
  Test_Determinant.cpp
  Test_DeterminantAndInverse.cpp
  Test_DotProduct.cpp
  Test_FrameTransform.cpp
  Test_Magnitude.cpp
  Test_Norms.cpp
  Test_OrthonormalOneform.cpp
  Test_OuterProduct.cpp
  Test_RaiseOrLowerIndex.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DataStructuresHelpers
  DomainHelpers
  ErrorHandling
  GeneralRelativity
  GeneralRelativityHelpers
  )

add_subdirectory(Python)
