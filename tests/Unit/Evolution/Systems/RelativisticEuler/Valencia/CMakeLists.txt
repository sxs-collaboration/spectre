# Distributed under the MIT License.
# See LICENSE.txt for details.

set(VALENCIA_DIR Evolution/Systems/RelativisticEuler/Valencia)

set(VALENCIA_TESTS
  ${VALENCIA_DIR}/Test_ConservativeFromPrimitive.cpp
  ${VALENCIA_DIR}/Test_Fluxes.cpp
  )

set(RELATIVISTIC_EULER_TESTS
  "${RELATIVISTIC_EULER_TESTS};${VALENCIA_TESTS}" PARENT_SCOPE
  )
set(LIBRARIES_TO_LINK "${LIBRARIES_TO_LINK};Valencia" PARENT_SCOPE)
