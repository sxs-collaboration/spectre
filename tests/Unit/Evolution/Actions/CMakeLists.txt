# Distributed under the MIT License.
# See LICENSE.txt for details.

set(EVOLUTION_ACTION_TESTS
  Evolution/Actions/Test_ComputeVolumeDuDt.cpp
  )

set(SPECTRE_TESTS "${SPECTRE_TESTS};${EVOLUTION_ACTION_TESTS}" PARENT_SCOPE)
