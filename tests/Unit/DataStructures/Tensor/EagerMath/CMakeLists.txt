# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_EagerMath")

set(LIBRARY_SOURCES
  Test_CrossProduct.cpp
  Test_Determinant.cpp
  Test_DeterminantAndInverse.cpp
  Test_DotProduct.cpp
  Test_Magnitude.cpp
  Test_Norms.cpp
  Test_OrthonormalOneform.cpp
  )

add_test_library(
  ${LIBRARY}
  "DataStructures/Tensor/EagerMath"
  "${LIBRARY_SOURCES}"
  ""
  )

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  DomainHelpers
  ErrorHandling
  GeneralRelativity
  )
