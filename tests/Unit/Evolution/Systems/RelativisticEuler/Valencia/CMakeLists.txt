# Distributed under the MIT License.
# See LICENSE.txt for details.

set(VALENCIA_TESTS
  Evolution/Systems/RelativisticEuler/Valencia/Test_ConservativeFromPrimitive.cpp
  )

set(RELATIVISTIC_EULER_TESTS
  "${RELATIVISTIC_EULER_TESTS};${VALENCIA_TESTS}" PARENT_SCOPE
  )
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};Valencia" PARENT_SCOPE)
